from __future__ import absolute_import, division, print_function

import glob
import os

import numpy as np
from PIL import Image as pil
from PIL import ImageFilter
try:
    import cv2
except Exception:
    cv2 = None

from .mono_dataset import MonoDataset


class C3VDDataset(MonoDataset):
    """C3VD monocular dataset loader (non-triplet split format).

    Expected split line format:
        <folder> <frame_idx> l
    """

    DEFAULT_K = np.array(
        [
            [0.56959306, 0.0, 0.5, 0.0],
            [0.0, 0.71185083, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    def __init__(self, *args, **kwargs):
        self.intrinsics_file = kwargs.pop("intrinsics_file", None)
        self.use_loss_mask = bool(kwargs.pop("use_loss_mask", False))
        self.mask_filename = str(kwargs.pop("mask_filename", "mask.png"))
        self.mask_erosion = int(kwargs.pop("mask_erosion", 0))
        super(C3VDDataset, self).__init__(*args, **kwargs)

        self.K = self.DEFAULT_K.copy()
        self._K_cache = {}
        self._path_cache = {}
        self._mask_path_cache = {}

    def check_depth(self):
        # Self-supervised path does not require GT depth in the dataloader.
        return False

    def check_loss_mask(self):
        return bool(self.use_loss_mask)

    def index_to_folder_and_frame_idx(self, index):
        parts = self.filenames[index].split()
        if len(parts) < 2:
            raise ValueError(
                "Invalid C3VD split line. Expected '<folder> <frame_idx> l', got: "
                f"{self.filenames[index]}"
            )

        folder = parts[0]
        frame_token = parts[1]
        side = parts[2] if len(parts) >= 3 else "l"

        try:
            frame_index = int(frame_token)
        except ValueError:
            frame_index = frame_token

        return folder, frame_index, side

    def _frame_tokens(self, frame_index):
        token = str(frame_index)
        tokens = [token]

        try:
            n = int(frame_index)
            for w in (3, 4, 5, 6, 7, 8, 9, 10):
                tokens.append(f"{n:0{w}d}")
        except (TypeError, ValueError):
            pass

        seen = set()
        unique = []
        for t in tokens:
            if t not in seen:
                unique.append(t)
                seen.add(t)
        return unique

    def _candidate_image_dirs(self, folder):
        return [
            folder,
            os.path.join(folder, "rgb"),
            os.path.join(folder, "images"),
            os.path.join(folder, "image"),
            os.path.join(folder, "color"),
            os.path.join(folder, "data"),
            os.path.join(folder, "left"),
            os.path.join(folder, "Left"),
            os.path.join(folder, "Left_rectified"),
        ]

    def _candidate_image_names(self, frame_index):
        exts = [self.img_ext, ".png", ".jpg", ".jpeg"]
        suffixes = ["", "_color", "-color"]

        names = []
        for token in self._frame_tokens(frame_index):
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

    def get_image_path(self, folder, frame_index, side):
        del side  # Monocular split token.

        cache_key = (folder, str(frame_index))
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        for rel_dir in self._candidate_image_dirs(folder):
            abs_dir = os.path.join(self.data_path, rel_dir)
            for name in self._candidate_image_names(frame_index):
                path = os.path.join(abs_dir, name)
                if os.path.isfile(path):
                    self._path_cache[cache_key] = path
                    return path

        # Last-resort recursive search for uncommon layouts.
        # Prefer explicit color files to avoid matching depth/flow/occlusion.
        root = os.path.join(self.data_path, folder)
        for token in self._frame_tokens(frame_index):
            for ext in ("png", "jpg", "jpeg"):
                for pat in (f"{token}_color.{ext}", f"{token}-color.{ext}", f"{token}.{ext}"):
                    hits = glob.glob(os.path.join(root, "**", pat), recursive=True)
                    if hits:
                        hits.sort()
                        self._path_cache[cache_key] = hits[0]
                        return hits[0]

                hits = glob.glob(
                    os.path.join(root, "**", f"{token}*.{ext}"),
                    recursive=True,
                )
                if hits:
                    hits.sort()
                    preferred = []
                    for h in hits:
                        stem = os.path.splitext(os.path.basename(h))[0].lower()
                        if stem == str(token).lower() or stem.endswith("_color") or stem.endswith("-color"):
                            preferred.append(h)
                    chosen = preferred[0] if preferred else hits[0]
                    self._path_cache[cache_key] = chosen
                    return chosen

        raise FileNotFoundError(
            f"C3VD frame not found for folder='{folder}', frame='{frame_index}' under {self.data_path}"
        )

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def _parse_intrinsics_file(self, path):
        ext = os.path.splitext(path)[1].lower()

        if ext in (".npy", ".npz"):
            arr = np.load(path)
            if isinstance(arr, np.lib.npyio.NpzFile):
                if not arr.files:
                    raise ValueError(f"Empty intrinsics npz file: {path}")
                arr = arr[arr.files[0]]
            M = np.asarray(arr, dtype=np.float32)
        else:
            rows = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace(",", " ")
                    vals = [float(x) for x in line.split()]
                    rows.append(vals)
            M = np.asarray(rows, dtype=np.float32)

        if M.ndim != 2:
            raise ValueError(f"Invalid intrinsics matrix shape in {path}: {M.shape}")

        if M.shape[0] >= 3 and M.shape[1] >= 3:
            fx = float(M[0, 0])
            fy = float(M[1, 1])
            cx = float(M[0, 2])
            cy = float(M[1, 2])
            return fx, fy, cx, cy

        raise ValueError(f"Invalid intrinsics matrix in {path}: {M.shape}")

    def _intrinsics_candidates(self, folder):
        candidates = []

        if self.intrinsics_file:
            candidates.append(self.intrinsics_file)

        folder_norm = folder.replace("\\", "/").strip("/")
        folder_root = folder_norm.split("/")[0] if folder_norm else folder_norm

        rel_candidates = [
            os.path.join(folder_norm, "intrinsics.txt"),
            os.path.join(folder_norm, "camera_intrinsics.txt"),
            os.path.join(folder_norm, "intrinsics.npy"),
            os.path.join(folder_root, "intrinsics.txt"),
            os.path.join(folder_root, "camera_intrinsics.txt"),
            os.path.join(folder_root, "intrinsics.npy"),
            "intrinsics.txt",
            "camera_intrinsics.txt",
            "intrinsics.npy",
        ]

        for rel in rel_candidates:
            if not rel:
                continue
            candidates.append(os.path.join(self.data_path, rel))

        normalized = []
        seen = set()
        for c in candidates:
            if not c:
                continue
            c_abs = c if os.path.isabs(c) else os.path.join(self.data_path, c)
            if c_abs not in seen:
                normalized.append(c_abs)
                seen.add(c_abs)
        return normalized

    def _native_image_size(self, folder, frame_index):
        path = self.get_image_path(folder, frame_index, "l")
        with pil.open(path) as img:
            w, h = img.size
        return float(w), float(h)

    def load_intrinsics(self, folder, frame_index):
        if folder in self._K_cache:
            return self._K_cache[folder].copy()

        K_norm = None
        for candidate in self._intrinsics_candidates(folder):
            if not os.path.isfile(candidate):
                continue
            try:
                fx, fy, cx, cy = self._parse_intrinsics_file(candidate)
            except Exception:
                continue

            pixel_like = (fx > 10.0) or (fy > 10.0) or (cx > 10.0) or (cy > 10.0)
            if pixel_like:
                w0, h0 = self._native_image_size(folder, frame_index)
                if w0 <= 0 or h0 <= 0:
                    continue
                fx /= w0
                fy /= h0
                cx /= w0
                cy /= h0

            K_norm = np.array(
                [
                    [fx, 0.0, cx, 0.0],
                    [0.0, fy, cy, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            break

        if K_norm is None:
            K_norm = self.DEFAULT_K.copy()

        self._K_cache[folder] = K_norm
        return K_norm.copy()

    def _mask_candidates(self, folder):
        folder_norm = folder.replace("\\", "/").strip("/")
        parts = [p for p in folder_norm.split("/") if p]

        candidates = []

        # Most-common locations: sequence root or sequence/masks.
        for i in range(len(parts), 0, -1):
            seq_rel = "/".join(parts[:i])
            candidates.append(os.path.join(self.data_path, seq_rel, self.mask_filename))
            candidates.append(os.path.join(self.data_path, seq_rel, "masks", self.mask_filename))
            candidates.append(os.path.join(self.data_path, seq_rel, "mask", self.mask_filename))

        candidates.append(os.path.join(self.data_path, folder_norm, self.mask_filename))
        candidates.append(os.path.join(self.data_path, self.mask_filename))

        seen = set()
        unique = []
        for c in candidates:
            if not c:
                continue
            if c not in seen:
                unique.append(c)
                seen.add(c)
        return unique

    def _resolve_mask_path(self, folder):
        if folder in self._mask_path_cache:
            return self._mask_path_cache[folder]

        for c in self._mask_candidates(folder):
            if os.path.isfile(c):
                self._mask_path_cache[folder] = c
                return c

        raise FileNotFoundError(
            f"C3VD mask file '{self.mask_filename}' not found for folder='{folder}' under {self.data_path}"
        )

    def get_loss_mask(self, folder, frame_index, side, do_flip):
        del frame_index, side

        mask_path = self._resolve_mask_path(folder)
        with pil.open(mask_path) as m:
            mask = m.convert("L")

        mask_np = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255

        if self.mask_erosion > 0:
            k = max(1, int(self.mask_erosion))
            if cv2 is not None:
                kernel = np.ones((2 * k + 1, 2 * k + 1), dtype=np.uint8)
                mask_np = cv2.erode(mask_np, kernel, iterations=1)
            else:
                size = max(3, 2 * k + 1)
                mask_np = np.asarray(
                    pil.fromarray(mask_np, mode="L").filter(ImageFilter.MinFilter(size)),
                    dtype=np.uint8,
                )

        mask_out = pil.fromarray(mask_np, mode="L")
        if do_flip:
            mask_out = mask_out.transpose(pil.FLIP_LEFT_RIGHT)
        return mask_out

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError("C3VD self-supervised training does not use depth in the dataloader.")
