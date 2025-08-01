# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import os

import argparse
import numpy as np
import PIL.Image as pil
import cv2

from utils import readlines
# from kitti_utils import generate_depth_map
from utils import readlines


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
                        choices=["endovis", "eigen", "eigen_benchmark","hamlyn","SERV-CT"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        #folder, frame_id, _ = line.split()
        #frame_id = int(frame_id)
        """
        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256"""
        print(line)
        folder, file,_ = line.split()
        folder = folder.split("/")[0]
        #line = line.replace("/","_")[1:]
        gt_depth_path = os.path.join(opt.data_path,"SERV-CT",folder,"Ground_truth_CT","DepthL", "{}.png".format(file))
        print(gt_depth_path)
        #gt_depth = np.array(pil.open(gt_depth_path))
        #im = pil.open(gt_depth_path)
        #newsize = (320, 256)
        #im = im.resize(newsize)
        im_gray = cv2.imread(gt_depth_path, cv2.IMREAD_GRAYSCALE)
        gt_depth = im_gray / 256

        print(gt_depth.shape)
        gt_depths.append(gt_depth)

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {} {}".format(opt.split,output_path))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
