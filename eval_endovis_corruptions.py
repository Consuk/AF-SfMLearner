# eval_endovis_corruptions.py
from __future__ import absolute_import, division, print_function
import os
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Reutilizamos los mismos módulos de tu script original
from options import MonodepthOptions
import datasets
import networks
from layers import disp_to_depth

# ===== Copiamos las mismas utilidades de métricas y helpers =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel  = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def load_model(load_weights_folder, num_layers, device):
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

def evaluate_one_root(data_path_root,
                      filenames,
                      gt_depths,
                      encoder,
                      depth_decoder,
                      height=256,
                      width=320,
                      batch_size=16,
                      num_workers=4,
                      png=False,
                      disable_median_scaling=False,
                      pred_depth_scale_factor=1.0,
                      device="cuda"):
    """
    Ejecuta la evaluación sobre un root (p.ej., .../brightness/severity_1/endovis_data)
    devolviendo el vector de métricas promedio.
    """
    # Verificación de que todos los archivos existen para esta raíz
    missing = [f for f in filenames if not os.path.isfile(os.path.join(data_path_root, f))]
    if len(missing) > 0:
        raise FileNotFoundError(
            f"[{data_path_root}] faltan {len(missing)} archivos de test_files; "
            f"primero faltante: {missing[0]}"
        )

    img_ext = '.png' if png else '.jpg'

    # Construimos el dataset con las mismas rutas relativas de filenames
    dataset = datasets.SCAREDRAWDataset(
        data_path_root, filenames, height, width, [0], 4, is_train=False, img_ext=img_ext
    )
    dataloader = DataLoader(dataset, batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, drop_last=False)

    pred_disps = []
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].to(device)
            features = encoder(input_color)
            output = depth_decoder(features)
            pred_disp, _ = disp_to_depth(output[("disp", 0)], 1e-3, 80)  # rangos como en monodepth
            pred_disps.append(pred_disp[:, 0].cpu().numpy())

    pred_disps = np.concatenate(pred_disps, axis=0)

    # Evaluamos con gt_depths ya cargado del split original (mismo orden)
    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):
        gt_depth = gt_depths[i]
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

    mean_errors = np.array(errors).mean(0)
    # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    return mean_errors

def main():
    parser = argparse.ArgumentParser("Evaluate EndoVIS corruptions (16x5) with AF-SfMLearner weights")
    parser.add_argument("--corruptions_root", type=str, required=True,
                        help="Raíz de las corrupciones (contiene carpetas: brightness, contrast, ...)")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Carpeta con encoder.pth y depth.pth")
    parser.add_argument("--split", type=str, default="endovis", help="Nombre del split (carpeta en ./splits/)")
    parser.add_argument("--splits_dir", type=str, default=os.path.join(os.path.dirname(__file__), "splits"))
    parser.add_argument("--num_layers", type=int, default=18)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--png", action="store_true")
    parser.add_argument("--eval_stereo", action="store_true", help="(Por si acaso) forzar estereo; default mono")
    parser.add_argument("--output_csv", type=str, default="corruptions_summary.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cargamos filenames y gts del split
    from utils import readlines
    test_files = readlines(os.path.join(args.splits_dir, args.split, "test_files.txt"))
    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró gt_depths.npz en {gt_path}")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    if len(test_files) != gt_depths.shape[0]:
        raise RuntimeError("test_files.txt y gt_depths.npz no tienen la misma longitud")

    # Configuración mono/estéreo como en tu script
    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    # Cargamos modelo una vez
    print("-> Cargando pesos:", args.load_weights_folder)
    encoder, depth_decoder = load_model(args.load_weights_folder, args.num_layers, device)

    # Detectamos las corrupciones (directorios 1er nivel)
    corr_types = sorted([d for d in os.listdir(args.corruptions_root)
                         if os.path.isdir(os.path.join(args.corruptions_root, d))])

    rows = []
    print("-> Iniciando evaluación de corrupciones")
    for corr in corr_types:
        corr_dir = os.path.join(args.corruptions_root, corr)
        # Detectamos severidades disponibles (severity_1 .. severity_5)
        severities = sorted([d for d in os.listdir(corr_dir)
                             if os.path.isdir(os.path.join(corr_dir, d)) and d.startswith("severity_")],
                            key=lambda s: int(s.split("_")[-1]))

        for sev in severities:
            data_root = os.path.join(corr_dir, sev, "endovis_data")
            print(f"\n>> {corr} / {sev} :: data_path = {data_root}")
            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            try:
                mean_errors = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    encoder=encoder,
                    depth_decoder=depth_decoder,
                    height=args.height,
                    width=args.width,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    png=args.png,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    device=device
                )
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append([corr, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3])

                print("   Métricas (promedio): "
                      f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                      f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}")

            except FileNotFoundError as e:
                print(f"   [SKIP] {e}")

    # Guardamos CSV
    if rows:
        import csv
        header = ["corruption", "severity", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

        print(f"\n-> Resumen guardado en: {args.output_csv}")

        # Pretty print por consola agrupado por corrupción
        from collections import defaultdict
        bucket = defaultdict(list)
        for r in rows:
            bucket[r[0]].append(r)

        print("\n======= RESUMEN (por corrupción) =======")
        for corr in sorted(bucket.keys()):
            print(f"\n{corr}")
            print("severity | abs_rel |  sq_rel |  rmse  | rmse_log |   a1   |   a2   |   a3")
            for _, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 in sorted(bucket[corr], key=lambda x: int(x[1].split('_')[-1])):
                print(f"{sev:>9} | {abs_rel:7.3f} | {sq_rel:7.3f} | {rmse:7.3f} |  {rmse_log:7.3f} | {a1:6.3f} | {a2:6.3f} | {a3:6.3f}")
    else:
        print("\n-> No se generaron filas. Revisa rutas/archivos faltantes.")

if __name__ == "__main__":
    # OpenCV single thread como en tu script
    cv2.setNumThreads(0)
    main()
