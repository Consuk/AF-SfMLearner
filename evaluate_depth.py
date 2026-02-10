from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

import datasets
import networks
from layers import disp_to_depth
from utils import readlines

import matplotlib.pyplot as plt

try:
    import wandb
except Exception:
    wandb = None

_DEPTH_COLORMAP = plt.get_cmap('plasma', 256)  # for plotting

cv2.setNumThreads(0)  # speeds up eval on unix systems

splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Stereo-trained models used nominal baseline 0.1; KITTI baseline is 54cm -> 5.4 scale
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths."""
    thresh = np.maximum((gt / pred), (pred / gt))

    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()

    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e-5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis


def build_eval_dataset(opt, filenames):
    """Creates the correct dataset for evaluation."""
    height = int(getattr(opt, "height", 256))
    width = int(getattr(opt, "width", 320))
    img_ext = ".png" if bool(getattr(opt, "png", False)) else ".jpg"

    if opt.eval_split == "hamlyn":
        DatasetClass = datasets.HamlynDataset
        print("-> Using HamlynDataset for evaluation")
    else:
        DatasetClass = datasets.SCAREDRAWDataset
        print(f"-> Using SCAREDRAWDataset for evaluation (eval_split={opt.eval_split})")

    dataset = DatasetClass(
        opt.data_path,
        filenames,
        height,
        width,
        [0],          # only current frame for eval
        4,            # num_scales
        is_train=False,
        img_ext=img_ext
    )
    return dataset


def evaluate(opt, global_step=None, log_to_wandb=True, max_log_images=3):
    """Evaluate a weights folder against gt_depths.npz.

    Returns:
        dict with keys: abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = float(getattr(opt, "max_depth", 150.0))

    assert sum((bool(getattr(opt, "eval_mono", False)), bool(getattr(opt, "eval_stereo", False)))) == 1, \
        "Choose mono or stereo evaluation: set exactly one of eval_mono/eval_stereo"

    if getattr(opt, "ext_disp_to_eval", None) is not None:
        print(f"-> Loading predictions from {opt.ext_disp_to_eval}")
        pred_disps = np.load(opt.ext_disp_to_eval)
    else:
        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
        assert os.path.isdir(opt.load_weights_folder), f"Cannot find folder: {opt.load_weights_folder}"

        print(f"-> Loading weights from {opt.load_weights_folder}")

        custom_list = getattr(opt, "eval_filelist", None)
        if custom_list is not None:
            custom_list = os.path.expanduser(custom_list)
            print(f"-> Using custom eval file list: {custom_list}")
            filenames = readlines(custom_list)
        else:
            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        dataset = build_eval_dataset(opt, filenames)

        batch_size = int(getattr(opt, "eval_batch_size", 16))
        dataloader = DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            num_workers=int(getattr(opt, "num_workers", 4)),
            pin_memory=True,
            drop_last=False
        )

        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path, map_location="cpu")

        encoder = networks.ResnetEncoder(int(getattr(opt, "num_layers", 18)), False)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(4))

        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

        encoder.cuda().eval()
        depth_decoder.cuda().eval()

        pred_disps = []
        height = int(getattr(opt, "height", 256))
        width = int(getattr(opt, "width", 320))
        print(f"-> Computing predictions with size {width}x{height}")

        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda(non_blocking=True)

                if bool(getattr(opt, "post_process", False)):
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                features = encoder(input_color)
                output = depth_decoder(features)

                pred_disp, _ = disp_to_depth(
                    output[("disp", 0)],
                    float(getattr(opt, "min_depth", 1e-3)),
                    float(getattr(opt, "max_depth", 150.0))
                )
                pred_disp = pred_disp.cpu()[:, 0].numpy()
                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps, axis=0)

    if bool(getattr(opt, "save_pred_disps", False)):
        output_path = os.path.join(opt.load_weights_folder, f"disps_{opt.eval_split}_split.npy")
        print(f"-> Saving predicted disparities to {output_path}")
        np.save(output_path, pred_disps)

    if bool(getattr(opt, "no_eval", False)):
        print("-> Evaluation disabled. Done.")
        return {}

    custom_gt = getattr(opt, "gt_depths_path", None)
    if custom_gt is not None:
        gt_path = os.path.expanduser(custom_gt)
        print(f"-> Using custom gt_depths.npz: {gt_path}")
    else:
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"gt_depths.npz not found: {gt_path}")

    data_npz = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)
    gt_depths = data_npz["data"]

    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    num_pred = pred_disps.shape[0]
    num_gt = len(gt_depths)
    assert num_pred == num_gt, f"Mismatch: {num_pred} predictions vs {num_gt} gt depth maps"

    if bool(getattr(opt, "eval_stereo", False)):
        print(f"   Stereo evaluation - scaling by {STEREO_SCALE_FACTOR}")
        disable_median_scaling = True
        pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")
        disable_median_scaling = bool(getattr(opt, "disable_median_scaling", False))
        pred_depth_scale_factor = float(getattr(opt, "pred_depth_scale_factor", 1.0))

    errors = []
    ratios = []

    can_wandb = log_to_wandb and (wandb is not None) and (wandb.run is not None)
    log_images = max(0, int(max_log_images))

    for i in range(num_pred):
        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp_resized = cv2.resize(pred_disp, (gt_width, gt_height), interpolation=cv2.INTER_LINEAR)

        pred_depth = 1.0 / np.maximum(pred_disp_resized, 1e-6)

        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask] * pred_depth_scale_factor
        gt = gt_depth[mask]

        if not disable_median_scaling:
            ratio = np.median(gt) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)
        errors.append(compute_errors(gt, pred_depth))

        if can_wandb and i < log_images:
            disp_vis = colormap(pred_disp_resized)
            wandb.log(
                {f"eval/disp_example_{i}": wandb.Image(disp_vis.transpose(1, 2, 0))},
                step=global_step if global_step is not None else i
            )

    if (not disable_median_scaling) and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(f" Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}")

    mean_errors = np.array(errors).mean(0)

    metrics = {
        "abs_rel": float(mean_errors[0]),
        "sq_rel": float(mean_errors[1]),
        "rmse": float(mean_errors[2]),
        "rmse_log": float(mean_errors[3]),
        "a1": float(mean_errors[4]),
        "a2": float(mean_errors[5]),
        "a3": float(mean_errors[6]),
    }

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("-> Done!")

    return metrics


if __name__ == "__main__":
    from options import MonodepthOptions
    options = MonodepthOptions()
    evaluate(options.parse())
