from __future__ import absolute_import, division, print_function

import os
import csv
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch

import datasets
import models

from utils import readlines
from layers import disp_to_depth


def compute_errors(gt, pred):
    """Standard depth metrics."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_gt_depths_npz(npz_path):
    gt_npz = np.load(npz_path, allow_pickle=True, fix_imports=True, encoding="latin1")
    gt_depths = gt_npz["data"]
    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)
    return gt_depths


def load_model(load_weights_folder, device, height=224, width=280):
    """
    Load EndoDAC model exactly like evaluate_depth.py logic.
    Expects depth_model.pth inside load_weights_folder.
    """
    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"No existe la carpeta de pesos: {load_weights_folder}")

    depth_model_path = os.path.join(load_weights_folder, "depth_model.pth")
    if not os.path.isfile(depth_model_path):
        raise FileNotFoundError(f"No existe {depth_model_path}")

    print(f"-> Loading EndoDAC weights from {load_weights_folder}")

    depther = models.endodac.endodac(
        backbone_size="base",
        r=0,
        image_shape=(height, width),
        lora_type="dvlora",
        pretrained_path=None,
        residual_block_indexes=[],
        include_cls_token=True,
    )

    model_dict = depther.state_dict()
    pretrained_dict = torch.load(depth_model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    depther.load_state_dict(model_dict)

    depther.to(device)
    depther.eval()
    return depther


def build_dataset(dataset_name, data_path_root, filenames, height, width, img_ext=".jpg"):
    """
    Build dataset following evaluate_depth.py logic.
    IMPORTANT: frame_idxs=[0] only, to avoid stereo/context frame loading.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == "hamlyn":
        return datasets.HamlynDataset(
            data_path_root,
            filenames,
            height,
            width,
            [0],
            4,
            is_train=False,
            img_ext=img_ext,
        )

    if dataset_name in ("endovis", "scared"):
        return datasets.SCAREDRAWDataset(
            data_path_root,
            filenames,
            height,
            width,
            [0],
            4,
            is_train=False,
            img_ext=img_ext,
        )

    if dataset_name == "c3vd":
        return datasets.C3VDDataset(
            data_path_root,
            filenames,
            height,
            width,
            [0],
            4,
            is_train=False,
            img_ext=img_ext,
        )

    raise ValueError(f"Dataset no soportado: {dataset_name}")


def evaluate_one_root(
    data_path_root,
    filenames,
    gt_depths,
    depther,
    dataset_name="hamlyn",
    height=224,
    width=280,
    batch_size=16,
    img_ext=".jpg",
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    min_depth=1e-3,
    max_depth=150.0,
    device="cuda",
):
    """
    Evaluate one corruption severity root.
    """
    dataset = build_dataset(
        dataset_name=dataset_name,
        data_path_root=data_path_root,
        filenames=filenames,
        height=height,
        width=width,
        img_ext=img_ext,
    )

    preds_list = []
    kept_indices = []

    buffer_imgs = []
    buffer_ids = []

    def flush_buffer():
        if not buffer_imgs:
            return
        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)
            output = depther(batch)
            pred_disp, _ = disp_to_depth(output[("disp", 0)], min_depth, max_depth)
            preds_list.append(pred_disp[:, 0].cpu().numpy())

    missing = 0
    total = len(filenames)
    debug_shown = 0

    for i in range(total):
        try:
            sample = dataset[i]
            img_t = sample[("color", 0, 0)]

            if not isinstance(img_t, torch.Tensor):
                img_t = torch.as_tensor(img_t)

            buffer_imgs.append(img_t)
            buffer_ids.append(i)

            if len(buffer_imgs) == batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except FileNotFoundError as e:
            missing += 1
            if debug_shown < 5:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' FileNotFoundError: {e}")
                debug_shown += 1
            if strict:
                raise FileNotFoundError(
                    f"[STRICT] Falta la muestra idx={i} en {data_path_root}: {e}"
                )
        except Exception as e:
            missing += 1
            if debug_shown < 5:
                print(f"   [DEBUG] idx={i} file='{filenames[i]}' error={repr(e)}")
                debug_shown += 1
            if strict:
                raise RuntimeError(
                    f"[STRICT] Error cargando idx={i} en {data_path_root}: {e}"
                )

    if buffer_imgs:
        flush_buffer()
        kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna muestra utilizable en {data_path_root}. "
            f"Faltantes/errores: {missing}/{total}"
        )

    if (not strict) and missing > 0:
        print(
            f"   [INFO] {data_path_root}: usando {len(kept_indices)}/{total} frames "
            f"(faltaron {missing})."
        )

    pred_disps = np.concatenate(preds_list, axis=0)

    if isinstance(gt_depths, list):
        sel_gt = [gt_depths[idx] for idx in kept_indices]
    else:
        sel_gt = gt_depths[kept_indices]

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):
        gt_depth = np.asarray(sel_gt[i]).astype(np.float32)
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)

        gd = gt_depth[mask]
        pd = pred_depth[mask]

        if pd.size == 0 or gd.size == 0:
            continue

        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gd) / (np.median(pd) + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < min_depth] = min_depth
        pd[pd > max_depth] = max_depth

        errors.append(compute_errors(gd, pd))

    if len(errors) == 0:
        raise RuntimeError(f"No se pudieron calcular métricas válidas en {data_path_root}")

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    return np.array(errors).mean(0)


def list_corruption_dirs(root):
    """
    If root already points to one corruption (contains severity_*), return [root].
    Otherwise, return immediate corruption subdirectories.
    """
    if not os.path.isdir(root):
        return []

    severities = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")
    ]
    if len(severities) > 0:
        return [root]

    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser("Evaluate corruptions using EndoDAC with evaluate_depth.py logic")

    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de corrupciones o carpeta de una corrupción")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con depth_model.pth")

    parser.add_argument("--splits_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "splits"),
                        help="Directorio base de splits")
    parser.add_argument("--split", type=str, default="hamlyn",
                        help="Nombre del split dentro de splits/")
    parser.add_argument("--dataset", type=str, default="hamlyn",
                        choices=["hamlyn", "endovis", "scared", "c3vd"],
                        help="Dataset a usar para construir el loader")
    parser.add_argument("--data_subdir", type=str, default="",
                        help="Subcarpeta dentro de severity_X para datasets no-Hamlyn")

    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=280)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_ext", type=str, default=".jpg")

    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--eval_stereo", action="store_true")

    parser.add_argument("--min_depth", type=float, default=1.0)
    parser.add_argument("--max_depth", type=float, default=50.0)

    parser.add_argument("--run_name", type=str, default="endodac_corruptions_eval")
    parser.add_argument("--output_dir", type=str, default="eval_outputs")
    parser.add_argument("--summary_filename", type=str, default="summary_by_severity.csv")
    parser.add_argument("--per_corruption_filename", type=str, default="summary_by_corruption.csv")
    parser.add_argument("--global_avg_filename", type=str, default="global_average.csv")

    args = parser.parse_args()

    cv2.setNumThreads(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_files_path = os.path.join(args.splits_dir, args.split, "test_files.txt")
    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")

    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"No se encontró: {test_files_path}")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró: {gt_path}")

    print(f"-> Using eval filelist: {test_files_path}")
    print(f"-> Using gt depths:    {gt_path}")

    test_files = readlines(test_files_path)
    gt_depths = load_gt_depths_npz(gt_path)

    if len(test_files) != len(gt_depths):
        print(
            "[WARN] test_files y gt_depths no tienen la misma longitud. "
            "El script seguirá y filtrará según las muestras realmente utilizables."
        )

    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = 1.0

    depther = load_model(
        load_weights_folder=args.load_weights_folder,
        device=device,
        height=args.height,
        width=args.width,
    )

    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No se encontraron corrupciones en {args.corruptions_root}")

    run_output_dir = os.path.join(args.output_dir, args.run_name)
    safe_makedirs(run_output_dir)

    rows = []

    print("-> Starting corruption evaluation")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))

        severities = sorted(
            [
                d for d in os.listdir(corr_dir)
                if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")
            ],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999
        )

        for sev in severities:
            if args.dataset.lower() == "hamlyn":
                data_root = os.path.join(corr_dir, sev)
            else:
                data_root = os.path.join(corr_dir, sev, args.data_subdir) if args.data_subdir else os.path.join(corr_dir, sev)

            print(f"\n>> {corr_name} / {sev} :: {data_root}")

            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            try:
                mean_errors = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    depther=depther,
                    dataset_name=args.dataset,
                    height=args.height,
                    width=args.width,
                    batch_size=args.batch_size,
                    img_ext=args.img_ext,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    strict=args.strict,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    device=device,
                )

                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                print(
                    f"   abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                    f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}"
                )

            except Exception as e:
                print(f"   [SKIP] {e}")

    if not rows:
        print("\n-> No se generaron resultados.")
        return

    header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    summary_csv = os.path.join(run_output_dir, args.summary_filename)
    save_csv(summary_csv, header, rows)
    print(f"\n-> CSV principal guardado en: {summary_csv}")

    bucket = defaultdict(list)
    for r in rows:
        bucket[r[0]].append(r)

    per_corr_rows = []
    for corr in sorted(bucket.keys()):
        vals = np.array([r[2:] for r in bucket[corr]], dtype=np.float64)
        means = vals.mean(axis=0).tolist()
        per_corr_rows.append([corr] + means)

    per_corr_header = ["corruption", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
    per_corr_csv = os.path.join(run_output_dir, args.per_corruption_filename)
    save_csv(per_corr_csv, per_corr_header, per_corr_rows)
    print(f"-> Promedio por corrupción guardado en: {per_corr_csv}")

    all_vals = np.array([r[2:] for r in rows], dtype=np.float64)
    global_means = all_vals.mean(axis=0).tolist()
    global_csv = os.path.join(run_output_dir, args.global_avg_filename)
    save_csv(global_csv, per_corr_header, [["global"] + global_means])
    print(f"-> Promedio global guardado en: {global_csv}")

    print("\n======= RESUMEN =======")
    print("Archivo principal:", summary_csv)
    print("Por corrupción   :", per_corr_csv)
    print("Global           :", global_csv)


if __name__ == "__main__":
    main()