from __future__ import absolute_import, division, print_function

import os
import argparse
import numpy as np
import cv2
import PIL.Image as pil

# Compatible import (tu repo suele tener utils.py en raíz)
try:
    from utils import readlines
except Exception:
    # fallback por si tienes estructura utils/utils.py
    from utils.utils import readlines


def export_gt_depths():
    parser = argparse.ArgumentParser(description="export_gt_depth")

    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the root of the data")

    parser.add_argument("--split", type=str, required=True,
                        choices=["eigen", "eigen_benchmark", "endovis", "hamlyn", "SERV-CT"],
                        help="which split to export gt from")

    # mantengo tu typo 'useage' para que tu comando funcione tal cual
    parser.add_argument("--useage", type=str, required=True,
                        choices=["eval", "3d_recon"],
                        help="gt depth use for evaluation or 3d reconstruction")

    parser.add_argument("--split_file_path", type=str, default=None,
                        help="optional path to a custom split file (e.g. test_files2.txt). "
                             "Overrides the default file in splits/<split>/")

    opt = parser.parse_args()

    # Decide split file + output
    if opt.split_file_path:
        split_file = opt.split_file_path
        base, _ = os.path.splitext(split_file)
        output_path = base + ("_gt_depths.npz" if opt.useage == "eval" else "_gt_depths_recon.npz")
    else:
        split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
        if opt.useage == "eval":
            split_file = os.path.join(split_folder, "test_files.txt")
            output_path = os.path.join(split_folder, "gt_depths.npz")
        else:
            split_file = os.path.join(split_folder, "3d_reconstruction.txt")
            output_path = os.path.join(split_folder, "gt_depths_recon.npz")

    lines = readlines(split_file)
    print(f"Exporting GT depths | split={opt.split} | useage={opt.useage} | N={len(lines)}")
    print(f"Split file: {split_file}")
    print(f"Output:    {output_path}")

    # Para fallbacks (si tu split file está fuera del repo)
    split_base_dir = os.path.dirname(opt.split_file_path) if opt.split_file_path else None

    gt_depths = []

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid line {i+1} in {split_file}: '{line}'")

        folder = parts[0]
        token2 = parts[1]
        side = parts[2] if len(parts) > 2 else None

        print(f"[{i+1:05d}] {folder} {token2} {side or ''}".rstrip())

        if opt.split == "eigen_benchmark":
            frame_id = int(token2)
            gt_depth_path = os.path.join(
                opt.data_path, folder, "proj_depth",
                "groundtruth", "image_02", "{:010d}.png".format(frame_id)
            )
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256.0

        elif opt.split == "endovis":
            frame_id = int(token2)
            f_str = "scene_points{:06d}.tiff".format(frame_id - 1)
            sequence = folder[7]
            data_splt = "train" if int(sequence) < 8 else "test"
            gt_depth_path = os.path.join(opt.data_path, data_splt, folder, "data", "scene_points", f_str)
            gt_depth = cv2.imread(gt_depth_path, 3)
            if gt_depth is None:
                raise FileNotFoundError(gt_depth_path)
            gt_depth = gt_depth[:, :, 0].astype(np.float32)
            gt_depth = gt_depth[0:1024, :]

        elif opt.split == "hamlyn":
            frame_id = int(token2)

            norm_folder = folder.strip("/")
            folder_parts = norm_folder.split("/")

            # si viene con image01/image02 -> lo cambiamos a depth01/depth02
            if any(p.startswith("image0") for p in folder_parts):
                parts_copy = []
                for p in folder_parts:
                    if p == "image01":
                        parts_copy.append("depth01")
                    elif p == "image02":
                        parts_copy.append("depth02")
                    else:
                        parts_copy.append(p)
                depth_base = os.path.join(*parts_copy)
            else:
                # si no viene imageXX, armamos <seq>/<seq>/depth01|depth02 según side
                if len(folder_parts) == 1:
                    seq_path = os.path.join(folder_parts[0], folder_parts[0])
                elif len(folder_parts) >= 2:
                    seq_path = os.path.join(folder_parts[-2], folder_parts[-1])
                else:
                    seq_path = norm_folder

                depth_sub = "depth02" if (side and side.lower().startswith("r")) else "depth01"
                depth_base = os.path.join(seq_path, depth_sub)

            fname = f"{frame_id:010d}"
            exts = [".tiff", ".tif", ".png", ".jpg", ".jpeg"]

            candidates = []
            for ext in exts:
                candidates.append(os.path.join(opt.data_path, depth_base, fname + ext))
            if split_base_dir:
                for ext in exts:
                    candidates.append(os.path.join(split_base_dir, depth_base, fname + ext))

            depth_path = next((p for p in candidates if os.path.isfile(p)), None)
            if depth_path is None:
                raise FileNotFoundError(
                    f"Could not find depth for {folder} frame={frame_id}. "
                    f"Tried base={os.path.join(opt.data_path, depth_base)}"
                )

            gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if gt_depth is None:
                raise RuntimeError(f"cv2.imread failed: {depth_path}")
            if gt_depth.ndim == 3:
                gt_depth = gt_depth[:, :, 0]
            gt_depth = gt_depth.astype(np.float32)

        elif opt.split == "SERV-CT":
            # conserva el patrón “viejo”: <folder> <file> ...
            file_name = token2
            gt_depth_path = os.path.join(opt.data_path, folder, file_name)
            gt_depth = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
            if gt_depth is None:
                raise FileNotFoundError(gt_depth_path)
            gt_depth = (gt_depth.astype(np.float32) / 256.0)[:256, :]

        else:
            raise ValueError(f"Unknown split: {opt.split}")

        gt_depths.append(gt_depth.astype(np.float32))

    # Hamlyn puede tener shapes distintas -> dtype=object
    gt_depths_arr = np.array(gt_depths, dtype=object)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, data=gt_depths_arr)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    export_gt_depths()