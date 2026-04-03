from __future__ import absolute_import, division, print_function

import argparse
import glob
import os

import numpy as np
try:
    import cv2
except Exception:
    cv2 = None

from PIL import Image

from utils import readlines, resolve_split_dir


def parse_split_line(line):
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(
            "Invalid split line. Expected '<folder> <frame_idx> l', got: "
            f"{line}"
        )
    folder = parts[0]
    frame_token = parts[1]
    side = parts[2] if len(parts) >= 3 else "l"
    return folder, frame_token, side


def frame_tokens(frame_token):
    tokens = [str(frame_token)]
    try:
        n = int(frame_token)
        for w in (3, 4, 5, 6, 7, 8, 9, 10):
            tokens.append(f"{n:0{w}d}")
    except (TypeError, ValueError):
        pass

    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def candidate_depth_dirs(folder):
    return [
        folder,
        os.path.join(folder, "depth"),
        os.path.join(folder, "depths"),
        os.path.join(folder, "depth_map"),
        os.path.join(folder, "depth_maps"),
        os.path.join(folder, "gt_depth"),
        os.path.join(folder, "gt_depths"),
        os.path.join(folder, "Ground_truth_CT", "DepthL"),
    ]


def candidate_depth_names(frame_token):
    names = []
    suffixes = ["", "_depth", "-depth"]
    exts = [".png", ".tif", ".tiff", ".npy"]
    for token in frame_tokens(frame_token):
        for suffix in suffixes:
            for ext in exts:
                names.append(f"{token}{suffix}{ext}")

    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            unique.append(n)
            seen.add(n)
    return unique


def find_c3vd_depth_file(data_path, folder, frame_token):
    for rel_dir in candidate_depth_dirs(folder):
        abs_dir = os.path.join(data_path, rel_dir)
        for name in candidate_depth_names(frame_token):
            path = os.path.join(abs_dir, name)
            if os.path.isfile(path):
                return path

    # Last-resort recursive search for uncommon layouts.
    root = os.path.join(data_path, folder)
    for token in frame_tokens(frame_token):
        for ext in ("png", "tif", "tiff", "npy"):
            hits = glob.glob(
                os.path.join(root, "**", f"{token}*.{ext}"),
                recursive=True,
            )
            if hits:
                hits.sort()
                return hits[0]

    return None


def decode_c3vd_depth(raw):
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype == np.uint16:
        depth_mm = arr.astype(np.float32) * (100.0 / 65535.0)
    else:
        depth_mm = arr.astype(np.float32)
    return np.clip(depth_mm, 0.0, 100.0)


def load_depth(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path))

    if cv2 is not None:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        raw = np.array(Image.open(path))

    if raw is None:
        raise RuntimeError(f"Could not read depth file: {path}")
    return raw


def export_gt_depths():
    parser = argparse.ArgumentParser(description="export_gt_depth")

    parser.add_argument("--data_path", type=str, required=True,
                        help="path to the root of the data")

    parser.add_argument("--split", type=str, required=True,
                        choices=["eigen", "eigen_benchmark", "endovis", "hamlyn", "SERV-CT", "c3vd"],
                        help="which split to export gt from")

    # Keep backward-compatible argument spelling used in this repo.
    parser.add_argument("--useage", type=str, default="eval",
                        choices=["eval", "3d_recon"],
                        help="gt depth use for evaluation or 3d reconstruction")

    parser.add_argument("--split_root", type=str, default=None,
                        help="optional root directory containing split folders (default: <repo>/splits)")
    parser.add_argument("--split_file_path", type=str, default=None,
                        help="optional path to a custom split file (e.g. test_files2.txt). "
                             "Overrides the default file in splits/<split>/")
    parser.add_argument("--allow_missing", action="store_true",
                        help="skip missing depth files instead of failing")

    opt = parser.parse_args()

    # Decide split file + output
    if opt.split_file_path:
        split_file = opt.split_file_path
        base, _ = os.path.splitext(split_file)
        output_path = base + ("_gt_depths.npz" if opt.useage == "eval" else "_gt_depths_recon.npz")
    else:
        split_root = opt.split_root or os.path.join(os.path.dirname(__file__), "splits")
        split_dir = resolve_split_dir(opt.split, split_root)
        if opt.useage == "eval":
            split_file = os.path.join(split_dir, "test_files.txt")
            output_path = os.path.join(split_dir, "gt_depths.npz")
        else:
            split_file = os.path.join(split_dir, "3d_reconstruction.txt")
            output_path = os.path.join(split_dir, "gt_depths_recon.npz")

    lines = readlines(split_file)
    print(f"Exporting GT depths | split={opt.split} | useage={opt.useage} | N={len(lines)}")
    print(f"Split file: {split_file}")
    print(f"Output:    {output_path}")

    split_base_dir = os.path.dirname(opt.split_file_path) if opt.split_file_path else None
    gt_depths = []
    missing = []

    for i, line in enumerate(lines):
        folder, frame_token, side = parse_split_line(line)
        frame_id = int(frame_token)

        print(f"[{i + 1:05d}] {folder} {frame_token} {side}")

        if opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(
                opt.data_path, folder, "proj_depth",
                "groundtruth", "image_02", f"{frame_id:010d}.png"
            )
            gt_depth = np.array(Image.open(gt_depth_path)).astype(np.float32) / 256.0

        elif opt.split == "endovis":
            f_str = f"scene_points{frame_id - 1:06d}.tiff"
            sequence = folder[7]
            data_splt = "train" if int(sequence) < 8 else "test"
            gt_depth_path = os.path.join(opt.data_path, data_splt, folder, "data", "scene_points", f_str)
            if cv2 is not None:
                gt_depth = cv2.imread(gt_depth_path, 3)
            else:
                gt_depth = np.array(Image.open(gt_depth_path))
            if gt_depth is None:
                missing.append((i, line))
                if not opt.allow_missing:
                    raise FileNotFoundError(gt_depth_path)
                continue
            gt_depth = gt_depth[:, :, 0].astype(np.float32)
            gt_depth = gt_depth[0:1024, :]

        elif opt.split == "hamlyn":
            norm_folder = folder.strip("/")
            folder_parts = norm_folder.split("/")

            # If line already points to image01/image02, switch to depth01/depth02.
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
                # Else build <seq>/<seq>/depth01|depth02 based on side.
                if len(folder_parts) == 1:
                    seq_path = os.path.join(folder_parts[0], folder_parts[0])
                elif len(folder_parts) >= 2:
                    seq_path = os.path.join(folder_parts[-2], folder_parts[-1])
                else:
                    seq_path = norm_folder

                depth_sub = "depth02" if (side and side.lower().startswith("r")) else "depth01"
                depth_base = os.path.join(seq_path, depth_sub)

            fname = f"{frame_id:010d}"
            exts = [".png", ".tiff", ".tif", ".jpg", ".jpeg"]
            candidates = []
            for ext in exts:
                candidates.append(os.path.join(opt.data_path, depth_base, fname + ext))
            if split_base_dir:
                for ext in exts:
                    candidates.append(os.path.join(split_base_dir, depth_base, fname + ext))

            depth_path = next((p for p in candidates if os.path.isfile(p)), None)
            if depth_path is None:
                missing.append((i, line))
                if not opt.allow_missing:
                    raise FileNotFoundError(
                        f"Could not find depth for {folder} frame={frame_id}. "
                        f"Tried base={os.path.join(opt.data_path, depth_base)}"
                    )
                continue

            raw = load_depth(depth_path)
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            gt_depth = raw.astype(np.float32)

        elif opt.split == "SERV-CT":
            gt_depth_path = os.path.join(opt.data_path, folder, frame_token)
            raw = load_depth(gt_depth_path)
            if raw.ndim == 3:
                raw = raw[:, :, 0]
            gt_depth = (raw.astype(np.float32) / 256.0)[:256, :]

        elif opt.split == "c3vd":
            depth_path = find_c3vd_depth_file(opt.data_path, folder, frame_token)
            if depth_path is None and split_base_dir is not None:
                depth_path = find_c3vd_depth_file(split_base_dir, folder, frame_token)

            if depth_path is None:
                missing.append((i, line))
                if not opt.allow_missing:
                    raise FileNotFoundError(
                        f"Depth file not found for line {i}: '{line}' under data_path={opt.data_path}"
                    )
                continue

            raw = load_depth(depth_path)
            gt_depth = decode_c3vd_depth(raw)

        else:
            raise ValueError(f"Unknown split: {opt.split}")

        gt_depths.append(gt_depth.astype(np.float32))

    if len(gt_depths) == 0:
        raise RuntimeError("No depth maps were exported.")

    gt_depths_arr = np.array(gt_depths, dtype=object)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(output_path, data=gt_depths_arr)
    print(f"Saved: {output_path}")
    print(f"Exported depth maps: {len(gt_depths)} / {len(lines)}")

    if missing:
        print(f"Missing entries: {len(missing)}")
        for idx, miss_line in missing[:10]:
            print(f"  - idx={idx}: {miss_line}")


if __name__ == "__main__":
    export_gt_depths()
