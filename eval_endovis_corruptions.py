# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function

import os
import argparse
import csv
import numpy as np
import cv2
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

# Reutilizamos utilidades y redes del repo
from options import MonodepthOptions  # solo para consistencia (no usado)
from utils import readlines
import networks
from layers import disp_to_depth

# Dataset EndoVIS (SCARED)
from datasets import SCAREDRAWDataset

try:
    from PIL import Image as PILImage
except Exception as e:
    raise ImportError("Pillow es requerido: pip install pillow") from e

# ===== Constantes/metas =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0


def compute_errors(gt, pred):
    """Métricas estándar de Monodepth/EndoDepth."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_model(load_weights_folder, num_layers, device):
    """Carga encoder.pth y depth.pth una sola vez."""
    encoder_path = os.path.join(load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(load_weights_folder, "depth.pth")

    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_weights_folder}")
    if not os.path.isfile(encoder_path) or not os.path.isfile(decoder_path):
        raise FileNotFoundError("Missing encoder.pth or depth.pth in weights folder")

    encoder = networks.ResnetEncoder(num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

    encoder_dict = torch.load(encoder_path, map_location=device)
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.to(device).eval()
    depth_decoder.to(device).eval()
    return encoder, depth_decoder


# ============================================================
#                 SPLIT PARSERS
# ============================================================
def _parse_endovis_split_line(line: str):
    """
    EndoVIS/SCARED típico tokenizado:
        dataset3 keyframe4 390 l
    Devuelve: ds, keyf, frame_idx:int, side:str
    """
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"Línea de split inválida: {line!r}")
    ds, keyf, frame_str, side = parts[0], parts[1], parts[2], parts[3]
    return ds, keyf, int(frame_str), side


def _parse_hamlyn_split_line(line: str):
    """
    Hamlyn (tu test_files2) suele ser:
        rectified05 0000000001 r
    o (a veces) con rel_path ya incluyendo image01/image02:
        rectified05/rectified05/image01 0000000001
        rectified05/rectified05/image02 0000000001

    Devuelve: rel_path:str, img_idx:str, side_or_cam:str|None
      - si hay 3 tokens: side_or_cam = 'l' o 'r'
      - si hay 2 tokens: side_or_cam = None (pero rel_path puede traer image01/image02)
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Línea Hamlyn inválida: {line!r}")
    rel_path = parts[0]
    img_idx = parts[1]
    side = parts[2] if len(parts) >= 3 else None
    return rel_path, img_idx, side


def _normalize_img_idx(img_idx: str) -> str:
    """Quita extensión y rellena a 10 dígitos si es numérico."""
    img_idx = os.path.splitext(img_idx)[0]
    if img_idx.isdigit() and len(img_idx) < 10:
        img_idx = img_idx.zfill(10)
    return img_idx


# ============================================================
#                 PATH BUILDERS
# ============================================================
def _build_endovis_img_path(root, ds, keyf, frame_idx, png=False):
    """<root>/<dataset>/<keyframe>/data/<frame>.<ext>"""
    ext = ".png" if png else ".jpg"
    return os.path.join(root, ds, keyf, "data", f"{frame_idx}{ext}")


def _find_hamlyn_img_path(sev_root: str, rel_path: str, img_idx: str, side: str, png: bool):
    """
    Encuentra la ruta de la imagen CORROMPIDA en un severity_root.

    Soporta:
      - rel_path = 'rectified05' + side -> busca:
          sev_root/rectified05/image01/img.jpg
          sev_root/rectified05/rectified05/image01/img.jpg   (TU CASO)
      - rel_path ya incluye image01/image02 -> busca:
          sev_root/<rel_path>/img.jpg
      - y variantes raras tipo rel_path='rectified05/image01'
          sev_root/rectified05/rectified05/image01/img.jpg
    """
    ext = ".png" if png else ".jpg"
    img_idx = _normalize_img_idx(img_idx)

    # Caso: rel_path ya trae image01/image02 al final
    if rel_path.endswith(("image01", "image02")):
        cand1 = os.path.join(sev_root, rel_path, f"{img_idx}{ext}")

        # fallback: rel_path tipo "rectified05/image01" pero real es doble-rectified
        cand2 = None
        pieces = rel_path.split("/")
        if len(pieces) == 2 and pieces[0].startswith("rectified") and pieces[1] in ("image01", "image02"):
            rect, cam = pieces[0], pieces[1]
            cand2 = os.path.join(sev_root, rect, rect, cam, f"{img_idx}{ext}")

        if os.path.isfile(cand1):
            return cand1
        if cand2 and os.path.isfile(cand2):
            return cand2
        return None

    # Caso: rel_path es solo la secuencia, se decide cam con side
    if side is None:
        # no hay l/r y tampoco image01/image02 incluido -> no podemos resolver
        return None

    cam = "image01" if side.lower().startswith("l") else "image02"

    cand1 = os.path.join(sev_root, rel_path, cam, f"{img_idx}{ext}")
    cand2 = os.path.join(sev_root, rel_path, rel_path, cam, f"{img_idx}{ext}")  # doble rectified

    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


# ============================================================
#                 EVALUATION CORES
# ============================================================
def evaluate_one_root_endovis(
    data_path_root,
    filenames,
    gt_depths,
    encoder,
    depth_decoder,
    height=256,
    width=320,
    batch_size=16,
    png=False,
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    device="cuda",
):
    """
    Evalúa EndoVIS/SCARED usando SCAREDRAWDataset y un bucle manual (lenient/strict).
    """
    img_ext = ".png" if png else ".jpg"
    try:
        dataset = SCAREDRAWDataset(
            data_path_root, filenames, height, width,
            [0], 4, is_train=False, img_ext=img_ext
        )
    except Exception as e:
        raise RuntimeError(f"No se pudo inicializar SCAREDRAWDataset en {data_path_root}: {e}")

    n = len(filenames)
    kept_indices = []
    preds_list = []

    buffer_imgs = []
    buffer_ids = []

    def flush_buffer():
        if len(buffer_imgs) == 0:
            return
        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)
            feats = encoder(batch)
            out = depth_decoder(feats)
            pred_disp, _ = disp_to_depth(out[("disp", 0)], 1e-3, 80)
            preds_list.append(pred_disp[:, 0].cpu().numpy())

    missing = 0
    for i in range(n):
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

        except FileNotFoundError:
            missing += 1
            if strict:
                raise FileNotFoundError(f"[STRICT] Falta la muestra del split idx={i} en {data_path_root}")
        except Exception as e:
            missing += 1
            if strict:
                raise RuntimeError(f"[STRICT] Error cargando idx={i} en {data_path_root}: {e}")

    flush_buffer()
    kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna imagen utilizable en {data_path_root} "
            f"(faltantes/errores: {missing}/{n})."
        )

    if (not strict) and missing > 0:
        print(f"   [INFO] {data_path_root}: usando {len(kept_indices)}/{n} frames del split (faltaron {missing}).")

    pred_disps = np.concatenate(preds_list, axis=0)
    sel_gt = gt_depths[kept_indices]

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = sel_gt[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pd = pred_depth[mask]
        gd = gt_depth[mask]

        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gd) / (np.median(pd) + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < MIN_DEPTH] = MIN_DEPTH
        pd[pd > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gd, pd))

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    return np.array(errors).mean(0)


def evaluate_one_root_hamlyn(
    sev_root,
    filenames,
    gt_depths,
    encoder,
    depth_decoder,
    height=256,
    width=320,
    batch_size=16,
    png=False,
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    device="cuda",
):
    """
    Evalúa Hamlyn CORROMPIDO leyendo rutas desde test_files2.txt.
    Espera que sev_root sea algo como:
      <corruption_dir>/severity_1
    y adentro se mantenga la estructura:
      rectifiedXX/rectifiedXX/image01/*.jpg
      rectifiedXX/rectifiedXX/image02/*.jpg
    """
    n = len(filenames)

    kept_indices = []
    preds_list = []

    buffer_imgs = []
    buffer_ids = []

    def flush_buffer():
        if len(buffer_imgs) == 0:
            return
        with torch.no_grad():
            batch = torch.stack(buffer_imgs, dim=0).to(device)
            feats = encoder(batch)
            out = depth_decoder(feats)
            pred_disp, _ = disp_to_depth(out[("disp", 0)], 1e-3, 80)
            preds_list.append(pred_disp[:, 0].cpu().numpy())

    missing = 0
    for i in range(n):
        line = filenames[i]
        try:
            rel_path, img_idx, side = _parse_hamlyn_split_line(line)
            img_idx = _normalize_img_idx(img_idx)

            img_path = _find_hamlyn_img_path(
                sev_root=sev_root,
                rel_path=rel_path,
                img_idx=img_idx,
                side=side,
                png=png,
            )
            if img_path is None or (not os.path.isfile(img_path)):
                missing += 1
                if strict:
                    raise FileNotFoundError(f"[STRICT] No encontrada idx={i}: {line.strip()} en {sev_root}")
                continue

            img = PILImage.open(img_path).convert("RGB")
            img = img.resize((width, height), PILImage.LANCZOS)
            img_np = (np.asarray(img).astype(np.float32) / 255.0)
            img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # C,H,W

            buffer_imgs.append(img_t)
            buffer_ids.append(i)

            if len(buffer_imgs) == batch_size:
                flush_buffer()
                kept_indices.extend(buffer_ids)
                buffer_imgs.clear()
                buffer_ids.clear()

        except Exception as e:
            missing += 1
            if strict:
                raise RuntimeError(f"[STRICT] Error idx={i} line={line.strip()}: {e}")

    flush_buffer()
    kept_indices.extend(buffer_ids)

    if len(kept_indices) == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna imagen utilizable en {sev_root} (faltantes/errores: {missing}/{n})."
        )

    if (not strict) and missing > 0:
        print(f"   [INFO] {sev_root}: usando {len(kept_indices)}/{n} frames del split (faltaron {missing}).")

    pred_disps = np.concatenate(preds_list, axis=0)
    sel_gt = gt_depths[kept_indices]

    errors, ratios = [], []
    for i in range(pred_disps.shape[0]):
        gt_depth = sel_gt[i]
        gt_h, gt_w = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_w, gt_h))
        pred_depth = 1.0 / (pred_disp + 1e-8)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        pd = pred_depth[mask]
        gd = gt_depth[mask]

        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        if not disable_median_scaling:
            ratio = np.median(gd) / (np.median(pd) + 1e-8)
            ratios.append(ratio)
            pd *= ratio

        pd[pd < MIN_DEPTH] = MIN_DEPTH
        pd[pd > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gd, pd))

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    return np.array(errors).mean(0)


# ============================================================
#                 DIR HELPERS
# ============================================================
def list_corruption_dirs(root):
    """
    Devuelve los directorios de primer nivel que representan corrupciones.
    Si 'root' ya es una carpeta de una corrupción (que contiene severity_*), la devuelve tal cual.
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
        os.path.join(root, d) for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def _load_split_and_gt(dataset_type, splits_dir, split_name, split_file, gt_depths_file):
    """
    Carga split+gt, soportando:
      - EndoVIS: test_files.txt y gt_depths.npz
      - Hamlyn : test_files2.txt y test_files2_gt_depths.npz
    Permite override con --split_file y --gt_depths_file.
    """
    if split_file is None:
        if dataset_type == "endovis":
            split_file = os.path.join(splits_dir, split_name, "test_files.txt")
        else:
            # hamlyn default
            split_file = os.path.join(splits_dir, "test_files2.txt") if os.path.isdir(splits_dir) and not os.path.isdir(os.path.join(splits_dir, split_name)) \
                else os.path.join(splits_dir, split_name, "test_files2.txt")

    if gt_depths_file is None:
        if dataset_type == "endovis":
            gt_depths_file = os.path.join(splits_dir, split_name, "gt_depths.npz")
        else:
            gt_depths_file = os.path.join(splits_dir, "test_files2_gt_depths.npz") if os.path.isdir(splits_dir) and not os.path.isdir(os.path.join(splits_dir, split_name)) \
                else os.path.join(splits_dir, split_name, "test_files2_gt_depths.npz")

    if not os.path.isfile(split_file):
        raise FileNotFoundError(f"No se encontró split_file: {split_file}")
    if not os.path.isfile(gt_depths_file):
        raise FileNotFoundError(f"No se encontró gt_depths_file: {gt_depths_file}")

    files = readlines(split_file)
    gt_depths = np.load(gt_depths_file, fix_imports=True, encoding="latin1")["data"]

    # Alinear longitudes para evitar out-of-bounds
    n_files = len(files)
    n_gt = gt_depths.shape[0]
    if n_files != n_gt:
        print(f"[WARN] split y gt_depths difieren: split={n_files}, gt={n_gt}. Se recorta a min().")
        n = min(n_files, n_gt)
        files = files[:n]
        gt_depths = gt_depths[:n]

    return files, gt_depths, split_file, gt_depths_file


def main():
    parser = argparse.ArgumentParser("Evaluate corruptions (EndoVIS or Hamlyn) with AF-SfMLearner weights")
    parser.add_argument("--dataset_type", type=str, choices=["endovis", "hamlyn"], default="endovis",
                        help="Tipo de dataset a evaluar (default: endovis).")

    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de las corrupciones (o una sola corrupción). Ej: /workspace/endovis_corruptions_test")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con encoder.pth y depth.pth")

    # EndoVIS-style splits_dir + split_name
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"),
                        help="Directorio donde viven los splits (para EndoVIS) o donde está test_files2 (Hamlyn).")
    parser.add_argument("--split", type=str, default="endovis",
                        help="Nombre del split (carpeta dentro de splits/) para EndoVIS. Para Hamlyn puedes dejarlo default.")

    # Overrides explícitos (recomendado para Hamlyn)
    parser.add_argument("--split_file", type=str, default=None,
                        help="Path explícito al split (override). Ej: /workspace/datasets/hamlyn/splits/test_files2.txt")
    parser.add_argument("--gt_depths_file", type=str, default=None,
                        help="Path explícito al gt_depths npz (override). Ej: /workspace/datasets/hamlyn/splits/test_files2_gt_depths.npz")

    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--png", action="store_true", help="Usa .png en lugar de .jpg (si tus corruptions son png).")
    parser.add_argument("--eval_stereo", action="store_true",
                        help="Forzar estéreo (desactiva median scaling y usa x5.4)")
    parser.add_argument("--output_csv", type=str, default="corruptions_summary.csv")
    parser.add_argument("--strict", action="store_true",
                        help="Modo estricto: exige que todas las entradas del split existan en cada severidad.")

    # EndoVIS subdir dentro de severity
    parser.add_argument("--endovis_data_subdir", type=str, default="endovis_data",
                        help="Solo EndoVIS: subdirectorio dentro de severity_*/ (default: endovis_data).")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2.setNumThreads(0)

    # Cargar split y GTs (según dataset_type)
    test_files, gt_depths, used_split, used_gt = _load_split_and_gt(
        dataset_type=args.dataset_type,
        splits_dir=args.splits_dir,
        split_name=args.split,
        split_file=args.split_file,
        gt_depths_file=args.gt_depths_file,
    )
    print("-> split_file:", used_split)
    print("-> gt_depths :", used_gt)
    print("-> #samples :", len(test_files))

    # Configuración mono/estéreo
    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    # Cargar modelo
    print("-> Cargando pesos:", args.load_weights_folder)
    encoder, depth_decoder = load_model(args.load_weights_folder, args.num_layers, device)

    # Detectar corrupciones
    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(f"No se encontraron carpetas de corrupción en {args.corruptions_root}")

    rows = []
    print("-> Iniciando evaluación de corrupciones")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))

        severities = sorted(
            [d for d in os.listdir(corr_dir)
             if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")],
            key=lambda s: int(s.split("_")[-1]) if s.split("_")[-1].isdigit() else 9999
        )

        for sev in severities:
            if args.dataset_type == "endovis":
                data_root = os.path.join(corr_dir, sev, args.endovis_data_subdir)
                pretty_root = data_root
            else:
                # Hamlyn: la raíz útil es el severity folder (adentro vive rectifiedXX/...)
                data_root = os.path.join(corr_dir, sev)
                pretty_root = data_root

            print(f"\n>> {corr_name} / {sev} :: data_path = {pretty_root}")
            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            try:
                if args.dataset_type == "endovis":
                    mean_errors = evaluate_one_root_endovis(
                        data_path_root=data_root,
                        filenames=test_files,
                        gt_depths=gt_depths,
                        encoder=encoder,
                        depth_decoder=depth_decoder,
                        height=args.height,
                        width=args.width,
                        batch_size=args.batch_size,
                        png=args.png,
                        disable_median_scaling=disable_median_scaling,
                        pred_depth_scale_factor=pred_depth_scale_factor,
                        strict=args.strict,
                        device=device,
                    )
                else:
                    mean_errors = evaluate_one_root_hamlyn(
                        sev_root=data_root,
                        filenames=test_files,
                        gt_depths=gt_depths,
                        encoder=encoder,
                        depth_decoder=depth_decoder,
                        height=args.height,
                        width=args.width,
                        batch_size=args.batch_size,
                        png=args.png,
                        disable_median_scaling=disable_median_scaling,
                        pred_depth_scale_factor=pred_depth_scale_factor,
                        strict=args.strict,
                        device=device,
                    )

                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                print("   Métricas (promedio): "
                      f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                      f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}")

            except FileNotFoundError as e:
                print(f"   [SKIP] {e}")
            except Exception as e:
                print(f"   [ERROR] {e}")

    # Guardar CSV y resumen
    if rows:
        header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

        print(f"\n-> Resumen guardado en: {args.output_csv}")

        bucket = defaultdict(list)
        for r in rows:
            bucket[r[0]].append(r)

        print("\n======= RESUMEN (por corrupción) =======")
        for corr in sorted(bucket.keys()):
            print(f"\n{corr}")
            print("severity | abs_rel |  sq_rel |  rmse  | rmse_log |   a1   |   a2   |   a3")
            for _, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 in sorted(
                bucket[corr],
                key=lambda x: int(x[1].split('_')[-1]) if x[1].split('_')[-1].isdigit() else 9999
            ):
                print(f"{sev:>9} | {abs_rel:7.3f} | {sq_rel:7.3f} | {rmse:7.3f} |  {rmse_log:7.3f} | "
                      f"{a1:6.3f} | {a2:6.3f} | {a3:6.3f}")
    else:
        print("\n-> No se generaron filas. Revisa rutas/archivos faltantes o estructura de corrupciones.")


if __name__ == "__main__":
    main()
