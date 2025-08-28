# export_gt_depth1.py  (versión sin cv2 + W&B)
import os
import argparse
import time
from os.path import join, exists, isdir, dirname

import numpy as np
from PIL import Image

try:
    import wandb
except Exception:
    wandb = None

from utils import readlines  # requiere 'six' instalado


DEPTH_DIR_CANDIDATES = ["DepthL", "depth", "depth_gt", "DepthR"]


def find_depth_dir(seq_root, frame_id):
    """Encuentra una subcarpeta de profundidad que tenga <frame_id>.png"""
    for d in DEPTH_DIR_CANDIDATES:
        p = join(seq_root, d, f"{frame_id}.png")
        if exists(p):
            return d
    return None


def load_depth_uint16(path):
    """Carga PNG 16-bit mono-canal y normaliza (mm/256 -> float32)."""
    im = Image.open(path)
    arr = np.array(im)
    if arr.ndim != 2 or arr.dtype != np.uint16:
        return None
    return arr.astype(np.float32) / 256.0


def export_gt_depths_endovis(opt):
    split_folder = join(dirname(__file__), "splits", opt.split)
    os.makedirs(split_folder, exist_ok=True)
    lines = readlines(join(split_folder, "test_files.txt"))

    run = None
    if opt.wandb and wandb is not None:
        run_name = opt.wandb_run or f"export_{opt.split}_{int(time.time())}"
        run = wandb.init(
            project=opt.wandb_project,
            entity=opt.wandb_entity,
            name=run_name,
            config=dict(
                split=opt.split,
                data_path=opt.data_path,
                resize=opt.resize,
                log_images_every=opt.wandb_log_images_every,
            ),
        )
        wandb.config.update({"script": "export_gt_depth1.py"}, allow_val_change=True)

    print(f"[export] split='{opt.split}'  lines={len(lines)}")
    gt_depths = []
    preview_records = []
    failures = 0

    last_seq = None
    cached_depth_dir = None

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            print(f"[WARN] Línea inesperada: '{line.strip()}'")
            continue

        folder = parts[0].split("/")[0]  # asegurar solo nombre raíz
        try:
            frame_id = int(parts[1])
        except Exception:
            print(f"[WARN] frame_id no entero en: '{line.strip()}'")
            continue

        seq_root = join(opt.data_path, folder)
        if not isdir(seq_root):
            print(f"[WARN] Secuencia no existe: {seq_root}")
            failures += 1
            continue

        # cachear la carpeta de depth por secuencia para acelerar
        if last_seq != seq_root:
            cached_depth_dir = find_depth_dir(seq_root, frame_id)
            last_seq = seq_root

        depth_dir = cached_depth_dir or find_depth_dir(seq_root, frame_id)
        if depth_dir is None:
            print(f"[WARN] Depth dir no encontrado para {folder}/{frame_id}.png")
            failures += 1
            continue

        depth_path = join(seq_root, depth_dir, f"{frame_id}.png")
        if not exists(depth_path):
            print(f"[WARN] Depth ausente: {depth_path}")
            failures += 1
            continue

        depth = load_depth_uint16(depth_path)
        if depth is None:
            print(f"[WARN] No es PNG uint16 mono-canal: {depth_path}")
            failures += 1
            continue

        if opt.resize:
            # NEAREST para no distorsionar valores de depth
            depth = np.array(
                Image.fromarray(depth).resize((320, 256), resample=Image.NEAREST),
                dtype=np.float32,
            )

        gt_depths.append(depth)

        # Loguear previews a W&B cada N
        if run is not None and opt.wandb_log_images_every > 0 and (idx % opt.wandb_log_images_every == 0):
            d = depth
            if np.any(d > 0):
                vmax = np.percentile(d[d > 0], 99.5)
            else:
                vmax = 1.0
            d_vis = np.clip((d / (vmax + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
            preview_records.append(
                {"folder": folder, "frame_id": frame_id, "depth_preview": d_vis}
            )

    out_npz = join(split_folder, "gt_depths.npz")
    np.savez_compressed(out_npz, data=np.array(gt_depths, dtype=np.float32))
    print(f"[export] saved {len(gt_depths)} depth maps -> {out_npz}  (failures={failures})")

    if run is not None:
        # Subir ejemplos como tabla (hasta 12)
        try:
            table = wandb.Table(columns=["folder", "frame_id", "depth_preview"])
            for rec in preview_records[:12]:
                table.add_data(rec["folder"], rec["frame_id"], wandb.Image(rec["depth_preview"]))
            if len(preview_records):
                wandb.log({"depth_examples": table})
        except Exception as e:
            print(f"[W&B] No se pudieron loguear ejemplos: {e}")

        # Artefacto con el NPZ
        try:
            art = wandb.Artifact(name=f"{opt.split}_gt_depths", type="dataset")
            art.add_file(out_npz, name="gt_depths.npz")
            run.log_artifact(art)
        except Exception as e:
            print(f"[W&B] No se pudo loguear artefacto: {e}")
        finally:
            run.finish()


def main():
    parser = argparse.ArgumentParser(description="export_gt_depth (endovis, sin cv2)")
    parser.add_argument('--data_path', type=str, required=True, help='Ruta raíz del dataset (endovis_data)')
    parser.add_argument('--split', type=str, required=True, choices=["endovis"], help='split a exportar')
    parser.add_argument('--resize', action='store_true', help='Redimensionar a 320x256 (ManyDepth/AF-SfM)')

    # W&B
    parser.add_argument('--wandb', action='store_true', help='Habilitar logging en Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='scarED', help='Proyecto W&B')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Entidad (usuario/equipo) W&B')
    parser.add_argument('--wandb_run', type=str, default=None, help='Nombre del run W&B')
    parser.add_argument('--wandb_log_images_every', type=int, default=500, help='Loguear preview cada N líneas (0=off)')

    opt = parser.parse_args()
    export_gt_depths_endovis(opt)


if __name__ == "__main__":
    main()
