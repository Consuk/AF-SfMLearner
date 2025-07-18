# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import os

import argparse
import numpy as np
import PIL.Image as pil
import torch
from os.path import exists
import cv2

from utils import readlines
# from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["endovis", "eigen", "eigen_benchmark","RNNSLAM"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    #newsize = (320, 256)
    for line in lines:
        #print(line)
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "endovis":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            
            if not exists(gt_depth_path):
                print(f"[WARNING] Depth file not found: {gt_depth_path}")
                continue

            # Leer la imagen como escala de grises
            im_gray = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)

            if im_gray is None:
                print(f"[WARNING] Could not read depth image: {gt_depth_path}")
                continue

            gt_depth = im_gray.astype(np.float32) / 256
            gt_depth = cv2.resize(gt_depth, (320, 256))

        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
        elif opt.split == "RNNSLAM":
            gt_depth_path = os.path.join(opt.data_path,folder, "{0}.png".format(frame_id))
            if(exists(gt_depth_path.replace("rgb","depth"))):
                gt_depth = np.array(pil.open(gt_depth_path.replace("rgb","depth"))).astype(np.float32) / 256
            else:
                gt_depth = np.array(pil.open(gt_depth_path.replace("rgb","depth_gt"))).astype(np.float32) / 256
            gt_depth = cv2.resize(gt_depth, (320, 256))
            #gt_depth = np.array(gt_depth.resize(newsize)).astype(np.float32) / 256

        gt_depths.append(gt_depth.astype(np.float32))
    """    
    gt_depth_tensor = torch.Tensor(gt_depths)
    median_ground_truth = torch.median(gt_depth_tensor)
    print(median_ground_truth)"""
    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()