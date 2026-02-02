from __future__ import absolute_import, division, print_function

import os
import numpy as np
from PIL import Image as pil

from datasets.mono_dataset import MonoDataset


class HamlynDataset(MonoDataset):
    """
    Hamlyn dataset loader with:
      - ENDO-DAC-style folder mapping: rectified08 -> rectified08/rectified08
      - Per-sequence intrinsics from intrinsics.txt (supports 3x3 or 3x4)
      - Split line parsing: <folder> <frame_index> <side>
        e.g. rectified08 1749 l
    """

    def __init__(self, *args, **kwargs):
        super(HamlynDataset, self).__init__(*args, **kwargs)

        self._K_cache = {}           # seq_dir_rel -> normalized K (4x4)
        self._actual_seq_cache = {}  # folder token -> actual sequence root (relative)

        self._side_to_cam = {
            "l": "image01",
            "r": "image02",
            "L": "image01",
            "R": "image02",
        }

        # These can be set from trainer after dataset construction
        self.strict_neighbors = getattr(self, "strict_neighbors", False)
        self.neighbor_search_max = int(getattr(self, "neighbor_search_max", 10))

    # -------------------------------------------------------------------------
    # Required by MonoDataset
    # -------------------------------------------------------------------------
    def check_depth(self):
        # Self-supervised training: no per-frame GT depth loaded here
        return False

    def index_to_folder_and_frame_idx(self, index):
        """
        Parse one line from split file.
        Expected: "<folder> <frame_index> <side>"
        Examples:
          "rectified08 1749 l"
          "rectified08/rectified08 1749 l"
          "rectified08/rectified08/image01 1749 l"
        """
        parts = self.filenames[index].split()
        folder = parts[0]
        frame_index = int(parts[1]) if len(parts) >= 2 else 0
        side = parts[2] if len(parts) >= 3 else "l"
        return folder, frame_index, side

    # -------------------------------------------------------------------------
    # ENDO-DAC-like folder mapping
    # -------------------------------------------------------------------------
    def _resolve_actual_sequence_root(self, folder):
        """
        ENDO-DAC style:
          - If folder is 'rectified08' but data exists at 'rectified08/rectified08', map it.
          - If folder already includes 'rectified08/rectified08', keep it.
          - Cache results for speed.
        Returns seq_dir_rel (relative to data_path), WITHOUT image01/image02.
        """
        norm = folder.replace("\\", "/").strip("/")
        if norm in self._actual_seq_cache:
            return self._actual_seq_cache[norm]

        cand1 = norm  # as provided
        base = norm.split("/")[0]
        cand2 = os.path.join(base, base).replace("\\", "/")  # rectified08/rectified08

        def looks_valid(seq_rel):
            intr = os.path.join(self.data_path, seq_rel, "intrinsics.txt")
            img1 = os.path.join(self.data_path, seq_rel, "image01")
            img2 = os.path.join(self.data_path, seq_rel, "image02")
            return os.path.exists(intr) or os.path.isdir(img1) or os.path.isdir(img2)

        if looks_valid(cand1):
            actual = cand1
        elif looks_valid(cand2):
            actual = cand2
        else:
            # fallback; will fail loudly later if wrong
            actual = cand1

        self._actual_seq_cache[norm] = actual
        return actual

    def _resolve_sequence_and_cam_dir(self, folder, side):
        """
        Supports:
          A) folder == rectified08
          B) folder == rectified08/rectified08
          C) folder == rectified08/rectified08/image01

        Returns:
          seq_dir_rel : e.g. rectified08/rectified08
          cam_dir     : image01 or image02
          folder_rel  : seq_dir_rel/cam_dir
        """
        norm = folder.replace("\\", "/").rstrip("/")

        # If split already includes camera folder
        if norm.endswith("/image01") or norm.endswith("/image02"):
            seq_dir_rel = os.path.dirname(norm)
            cam_dir = os.path.basename(norm)
            seq_dir_rel = self._resolve_actual_sequence_root(seq_dir_rel)
            folder_rel = os.path.join(seq_dir_rel, cam_dir).replace("\\", "/")
            return seq_dir_rel, cam_dir, folder_rel

        cam_dir = self._side_to_cam.get(side, "image01")
        seq_dir_rel = self._resolve_actual_sequence_root(norm)
        folder_rel = os.path.join(seq_dir_rel, cam_dir).replace("\\", "/")
        return seq_dir_rel, cam_dir, folder_rel

    # -------------------------------------------------------------------------
    # Image loading
    # -------------------------------------------------------------------------
    def get_image_path(self, folder, frame_index, side):
        _, _, folder_rel = self._resolve_sequence_and_cam_dir(folder, side)
        frame_str = f"{int(frame_index):010d}"
        return os.path.join(self.data_path, folder_rel, frame_str + self.img_ext)

    def get_color(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        color = self.loader(path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    # -------------------------------------------------------------------------
    # Intrinsics per sequence (intrinsics.txt in seq root)
    # -------------------------------------------------------------------------
    def _read_intrinsics_txt(self, intr_path):
        """
        Supports:
          - 3 lines of 3 values (3x3)
          - 3 lines of 4 values (3x4)
          - or any line format where first 3 rows have >=3 entries
        Extracts fx, fy, cx, cy from the matrix.
        """
        rows = []
        with open(intr_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = [float(x) for x in line.split()]
                rows.append(vals)

        if len(rows) < 3:
            raise ValueError(f"Invalid intrinsics file (need >=3 rows): {intr_path}")

        M = np.zeros((3, 4), dtype=np.float32)
        for r in range(3):
            row = rows[r]
            if len(row) >= 4:
                M[r, :] = row[:4]
            elif len(row) == 3:
                M[r, :3] = row
            else:
                raise ValueError(f"Row {r} has too few columns in {intr_path}: {row}")

        fx = float(M[0, 0])
        fy = float(M[1, 1])
        cx = float(M[0, 2])
        cy = float(M[1, 2])
        return fx, fy, cx, cy

    def _get_native_image_size(self, folder, frame_index):
        # use left camera by default
        path = self.get_image_path(folder, frame_index, "l")
        with pil.open(path) as img:
            w, h = img.size
        return w, h

    def load_intrinsics(self, folder, frame_index):
        """
        Returns normalized 4x4 K.
        MonoDataset will scale by width/height per pyramid scale.
        """
        seq_dir_rel = self._resolve_actual_sequence_root(folder)
        if seq_dir_rel in self._K_cache:
            return self._K_cache[seq_dir_rel].copy()

        intr_path = os.path.join(self.data_path, seq_dir_rel, "intrinsics.txt")
        if not os.path.exists(intr_path):
            raise FileNotFoundError(
                f"Could not find intrinsics.txt for '{seq_dir_rel}'. Expected: {intr_path}"
            )

        fx, fy, cx, cy = self._read_intrinsics_txt(intr_path)

        # Heuristic: if values look like pixels, normalize by native size
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
        # Not used for self-supervised training; evaluation uses gt_depths.npz
        raise NotImplementedError("Self-supervised Hamlyn training: no per-frame GT depth used here.")
