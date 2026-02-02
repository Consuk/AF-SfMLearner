from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image as pil

from datasets.mono_dataset import MonoDataset


class HamlynDataset(MonoDataset):
    """
    Hamlyn dataset loader for Monodepth2-style training with per-folder intrinsics.

    Expected Hamlyn structure (common):
      Hamlyn/
        rectified15/
          rectified15/
            intrinsics.txt
            image01/
              0000000001.jpg
              ...
            image02/
              0000000001.jpg
              ...

    Split lines can be either:
      1) folder includes camera folder already:
            rectified15/rectified15/image01 1 l
      2) folder points to the sequence root (no image01/image02):
            rectified15/rectified15 1 l

    This loader supports both.

    Intrinsics:
      Reads intrinsics.txt in each sequence folder.
      If values look like pixels, normalize by the native (w,h) of that sequence.
      If values already look normalized, keep as-is.
    """

    def __init__(self, *args, **kwargs):
        super(HamlynDataset, self).__init__(*args, **kwargs)

        self._K_cache = {}  # seq_dir_rel -> normalized K_4x4

        # optional strict neighbor snapping (set by Trainer)
        self.strict_neighbors = getattr(self, "strict_neighbors", False)
        self.neighbor_search_max = getattr(self, "neighbor_search_max", 10)

    def _read_intrinsics_txt(self, intr_path):
        with open(intr_path, "r") as f:
            parts = f.read().strip().split()
        vals = list(map(float, parts))
        if len(vals) != 4:
            raise ValueError(
                f"Expected 4 numbers in intrinsics.txt (fx fy cx cy), got {len(vals)} in {intr_path}"
            )
        fx, fy, cx, cy = vals
        return fx, fy, cx, cy

    def _resolve_actual_sequence_root(self, folder):
        """
        Resolve actual path to the sequence folder containing intrinsics.txt, given a split 'folder' token.
        We try a few common layouts:
          - folder itself already equals "<rectifiedXX>/<rectifiedXX>"
          - folder equals "<rectifiedXX>" and needs "<rectifiedXX>/<rectifiedXX>"
        """
        folder = folder.strip("/")

        def looks_valid(rel):
            return os.path.exists(os.path.join(self.data_path, rel, "intrinsics.txt"))

        if looks_valid(folder):
            return folder

        base = folder.split("/")[0]
        cand2 = os.path.join(base, base)
        if looks_valid(cand2):
            return cand2

        raise FileNotFoundError(
            f"Could not resolve sequence root for folder='{folder}'. "
            f"Tried '{folder}' and '{cand2}'."
        )

    def _resolve_sequence_and_cam_dir(self, folder, side):
        """
        Given folder from split and side ('l'/'r'), return:
          seq_dir_rel: "<rectifiedXX>/<rectifiedXX>"
          cam_dir: "image01" or "image02"
          frame_prefix_dir: optional if folder already includes cam dir
        """
        folder = folder.strip("/")

        # If folder already ends in image01/image02, treat parent as seq root
        norm = folder.replace("\\", "/")
        tail = norm.split("/")[-1].lower()
        if tail in ["image01", "image02"]:
            seq_dir_rel = self._resolve_actual_sequence_root("/".join(norm.split("/")[:-1]))
            cam_dir = tail
            return seq_dir_rel, cam_dir, norm  # norm is full path including cam dir

        # Else folder is sequence root
        seq_dir_rel = self._resolve_actual_sequence_root(folder)
        cam_dir = "image01" if side == "l" else "image02"
        return seq_dir_rel, cam_dir, os.path.join(seq_dir_rel, cam_dir)

    def _get_native_image_size(self, folder, frame_index):
        """
        Read one image (frame_index) to determine native (w,h) for normalization.
        """
        # assume left camera path for size (same for both)
        seq_dir_rel, cam_dir, frame_prefix = self._resolve_sequence_and_cam_dir(folder, "l")
        img_path = os.path.join(self.data_path, frame_prefix, f"{int(frame_index):010d}{self.img_ext}")
        with pil.open(img_path) as im:
            w, h = im.size
        return w, h

    def load_intrinsics(self, folder, frame_index):
        """
        Returns normalized 4x4 K. Monodepth2 will scale K by (width,height)
        inside MonoDataset.__getitem__.
        """
        # folder may already include /image01 or /image02 - strip it before looking for intrinsics
        seq_dir_rel, _, _ = self._resolve_sequence_and_cam_dir(folder, "l")
        if seq_dir_rel in self._K_cache:
            return self._K_cache[seq_dir_rel].copy()

        intr_path = os.path.join(self.data_path, seq_dir_rel, "intrinsics.txt")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(
                f"Could not find intrinsics.txt for '{seq_dir_rel}'. "
                f"Expected: {intr_path}"
            )

        fx, fy, cx, cy = self._read_intrinsics_txt(intr_path)

        # pixel-like heuristic
        pixel_like = (fx > 10.0) or (fy > 10.0) or (cx > 10.0) or (cy > 10.0)
        if pixel_like:
            w0, h0 = self._get_native_image_size(folder, frame_index)
            fx_n = fx / float(w0)
            fy_n = fy / float(h0)
            cx_n = cx / float(w0)
            cy_n = cy / float(h0)
        else:
            fx_n, fy_n, cx_n, cy_n = fx, fy, cx, cy

        K_norm = np.array([
            [fx_n, 0.0,  cx_n, 0.0],
            [0.0,  fy_n, cy_n, 0.0],
            [0.0,  0.0,  1.0,  0.0],
            [0.0,  0.0,  0.0,  1.0]
        ], dtype=np.float32)

        self._K_cache[seq_dir_rel] = K_norm.copy()
        return K_norm.copy()

    def get_depth(self, folder, frame_index, side, do_flip):
        """
        Hamlyn has GT depth only for evaluation (gt_depths.npz). For training self-supervised,
        we do not load per-frame depths here.
        """
        raise NotImplementedError("HamlynDataset.get_depth is not used for training; use gt_depths.npz for evaluation.")

    def check_depth(self):
        # Self-supervised training does not require depth files
        return False

    def get_image_path(self, folder, frame_index, side):
        """
        Return absolute image file path.
        Supports both split folder styles (with/without image01/image02).
        """
        side = side.lower()
        seq_dir_rel, cam_dir, frame_prefix = self._resolve_sequence_and_cam_dir(folder, side)
        fname = f"{int(frame_index):010d}{self.img_ext}"
        return os.path.join(self.data_path, frame_prefix, fname)
