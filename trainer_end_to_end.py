from __future__ import absolute_import, division, print_function

import time
import json
import os
import datasets
import networks
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.cm as cm


from utils import *
from layers import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import gc
import torch

try:
    import wandb
except Exception:
    wandb = None

class Trainer:
    def _init_wandb(self):
        self.wandb_enabled = False
        if getattr(self.opt, "use_wandb", False) and (wandb is not None):
            run_name = getattr(self.opt, "wandb_run_name", None) or self.opt.model_name
            kwargs = dict(
                project=getattr(self.opt, "wandb_project", "af-sfmlearner"),
                name=run_name,
                config=self.opt.__dict__,
            )
            entity = getattr(self.opt, "wandb_entity", None)
            if entity:
                kwargs["entity"] = entity
            # Inicializa run
            wandb.init(**kwargs)
            self.wandb_enabled = True
            print("-> W&B enabled")
        else:
            print("-> W&B disabled")

    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
        
        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "height must be a multiple of 32"
        assert self.opt.width % 32 == 0, "width must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"
        
        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.num_input_frames = len(self.opt.frame_ids)

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["transform_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained",
                                                                 num_input_images=self.num_pose_frames)
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.PoseDecoder(self.models["transform_encoder"].num_ch_enc,
                                                       num_input_features=1,
                                                       num_frames_to_predict_for=2)
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        self.models["position_encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained",
                                                                 num_input_images=self.num_pose_frames)
        self.models["position_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["position_encoder"].parameters())

        self.models["position"] = networks.PoseDecoder(self.models["position_encoder"].num_ch_enc,
                                                       num_input_features=1,
                                                       num_frames_to_predict_for=2)
        self.models["position"].to(self.device)
        self.parameters_to_train += list(self.models["position"].parameters())

        if self.opt.predictive_mask:
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, 
                self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids)-1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        if self.opt.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = networks.ResnetEncoder(
                self.opt.num_layers,
                self.opt.weights_init == "pretrained",
                num_input_images=self.num_pose_frames)

            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = networks.PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2)

        elif self.opt.pose_model_type == "shared":
            self.models["pose"] = networks.PoseDecoder(
                self.models["encoder"].num_ch_enc, num_input_features=2,
                num_frames_to_predict_for=2)

        elif self.opt.pose_model_type == "posecnn":
            self.models["pose"] = networks.PoseCNN(
                self.num_input_frames if self.opt.pose_model_input == "all" else 2)

        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        self.print_params()

        self.num_scales = len(self.opt.scales)

        self.min_depth = self.opt.min_depth
        self.max_depth = self.opt.max_depth

        # data
        datasets_dict = {"endovis": datasets.SCAREDRAWDataset, "hamlyn": datasets.HamlynDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        train_filenames = readlines(fpath.format("train"))

        # Some Hamlyn splits don't ship a val_files.txt. If it's missing, fall back to test_files.txt.
        try:
            val_filenames = readlines(fpath.format("val"))
            if len(val_filenames) == 0:
                raise FileNotFoundError("val_files.txt is empty")
        except Exception:
            test_fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "test_files.txt")
            if os.path.exists(test_fpath):
                print(f"[WARN] No valid val_files.txt found for split '{self.opt.split}'. Using test_files.txt as validation.")
                val_filenames = readlines(test_fpath)
            else:
                print(f"[WARN] No valid val_files.txt or test_files.txt found for split '{self.opt.split}'. Using train as validation (NOT recommended).")
                val_filenames = train_filenames

        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)

        # Hamlyn ENDO-DAC-style neighbor snapping (optional)
        if self.opt.dataset == "hamlyn":
            train_dataset.strict_neighbors = bool(getattr(self.opt, "hamlyn_strict_neighbors", False))
            train_dataset.neighbor_search_max = int(getattr(self.opt, "neighbor_search_max", 10))

        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)

        if self.opt.dataset == "hamlyn":
            val_dataset.strict_neighbors = bool(getattr(self.opt, "hamlyn_strict_neighbors", False))
            val_dataset.neighbor_search_max = int(getattr(self.opt, "neighbor_search_max", 10))

        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, False,
            num_workers=1, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # keep references for debugging / logging
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        self._init_wandb()

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.spatial_transform = SpatialTransformer((self.opt.height, self.opt.width))
        self.spatial_transform.to(self.device)

        self.get_occu_mask_backward = get_occu_mask_backward((self.opt.height, self.opt.width))
        self.get_occu_mask_backward.to(self.device)

        self.get_occu_mask_bidirection = get_occu_mask_bidirection((self.opt.height, self.opt.width))
        self.get_occu_mask_bidirection.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        self.position_depth = {}
        
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

            self.position_depth[scale] = optical_flow((h, w), self.opt.batch_size, h, w)
            self.position_depth[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        if self.opt.dataset == "hamlyn" and bool(getattr(self.opt, "hamlyn_strict_neighbors", False)):
            print(f"-> Hamlyn strict neighbor snapping is ENABLED (neighbor_search_max={train_dataset.neighbor_search_max})")

        self.save_opts()

    def set_train_0(self):
        """Convert all models to training mode
        """
        for param in self.models["position_encoder"].parameters():
            param.requires_grad = True
        for param in self.models["position"].parameters():
            param.requires_grad = True

        for param in self.models["encoder"].parameters():
            param.requires_grad = False
        for param in self.models["depth"].parameters():
            param.requires_grad = False
        
        self.models["position_encoder"].train()
        self.models["position"].train()
        
        self.models["encoder"].eval()
        self.models["depth"].eval()

    def set_train_1(self):
        """Convert all models to training mode
        """
        for param in self.models["position_encoder"].parameters():
            param.requires_grad = False
        for param in self.models["position"].parameters():
            param.requires_grad = False

        for param in self.models["encoder"].parameters():
            param.requires_grad = True
        for param in self.models["depth"].parameters():
            param.requires_grad = True
        
        self.models["position_encoder"].eval()
        self.models["position"].eval()

        self.models["encoder"].train()
        self.models["depth"].train()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()

        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            # Always save on schedule, and ensure weights exist if we evaluate each epoch.
            saved_this_epoch = False
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
                saved_this_epoch = True

            if bool(getattr(self.opt, "eval_each_epoch", False)):
                if not saved_this_epoch:
                    # We need a weights folder to evaluate against.
                    self.save_model()
                weights_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch}")
                self.evaluate_each_epoch_if_enabled(weights_folder)

    def evaluate_each_epoch_if_enabled(self, weights_folder):
        """Runs depth evaluation using gt_depths.npz (e.g., Hamlyn) and logs metrics."""
        if not bool(getattr(self.opt, "eval_each_epoch", False)):
            return

        # Quick existence checks (avoid crashing mid-training)
        splits_dir = os.path.join(os.path.dirname(__file__), "splits")
        gt_path = os.path.join(splits_dir, self.opt.eval_split, "gt_depths.npz")
        test_path = os.path.join(splits_dir, self.opt.eval_split, "test_files.txt")
        if not os.path.exists(gt_path):
            print(f"[WARN] eval_each_epoch is set but gt file not found: {gt_path}. Skipping evaluation.")
            return
        if not os.path.exists(test_path):
            print(f"[WARN] eval_each_epoch is set but test_files.txt not found: {test_path}. Skipping evaluation.")
            return

        try:
            import evaluate_depth as eval_depth
        except Exception as e:
            print(f"[WARN] Could not import evaluate_depth.py for evaluation: {e}. Skipping evaluation.")
            return

        from types import SimpleNamespace

        eval_opt = SimpleNamespace(
            eval_mono=True,
            eval_stereo=False,
            disable_median_scaling=bool(getattr(self.opt, "disable_median_scaling", False)),
            pred_depth_scale_factor=float(getattr(self.opt, "pred_depth_scale_factor", 1.0)),
            ext_disp_to_eval=None,
            load_weights_folder=weights_folder,
            num_layers=int(getattr(self.opt, "num_layers", 18)),
            post_process=False,
            min_depth=float(getattr(self.opt, "min_depth", 1e-3)),
            max_depth=float(getattr(self.opt, "max_depth", 150.0)),
            data_path=self.opt.data_path,
            eval_split=self.opt.eval_split,
            png=bool(getattr(self.opt, "png", False)),
            num_workers=int(getattr(self.opt, "num_workers", 4)),
            eval_batch_size=int(getattr(self.opt, "eval_batch_size", 16)),
            save_pred_disps=False,
            no_eval=False,
            eval_eigen_to_benchmark=False,
            eval_out_dir=None,
            height=int(getattr(self.opt, "height", 256)),
            width=int(getattr(self.opt, "width", 320)),
        )

        metrics = eval_depth.evaluate(
            eval_opt,
            global_step=self.step,
            log_to_wandb=bool(getattr(self, "wandb_enabled", False)),
            max_log_images=int(getattr(self.opt, "wandb_eval_max_images", 3))
        )

        if not metrics:
            return

        # TensorBoard
        try:
            writer = self.writers.get("val", None)
            if writer is not None:
                for k, v in metrics.items():
                    writer.add_scalar(f"eval/{k}", v, self.step)
        except Exception:
            pass

        # W&B (same run)
        if bool(getattr(self, "wandb_enabled", False)):
            try:
                wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.step)
            except Exception:
                pass

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if self.step % 400 == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

            if self.step % self.opt.log_frequency == 0:
                self.log("train", inputs, outputs, losses)

                self.val()
            
            self.step += 1
            
        if self.wandb_enabled:
            wandb.log({"epoch": self.epoch}, step=self.step)


    def _deep_feat(self, feats):
        """Return the deepest feature map tensor from a ResNet encoder output."""
        if isinstance(feats, (list, tuple)):
            return feats[-1]
        return feats

    def predict_poses(self, inputs):
        """Predict poses between target (0) and each source frame in frame_ids (Monodepth2-style)."""
        outputs = {}

        for f_i in self.opt.frame_ids[1:]:
            if f_i == "s":
                continue

            # PoseCNN path
            if self.opt.pose_model_type == "posecnn":
                if self.opt.pose_model_input == "all":
                    pose_in = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids], 1)
                else:
                    if f_i < 0:
                        pose_in = torch.cat([inputs[("color_aug", f_i, 0)], inputs[("color_aug", 0, 0)]], 1)
                    else:
                        pose_in = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", f_i, 0)]], 1)

                axisangle, translation = self.models["pose"](pose_in)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=bool(f_i < 0)
                )
                continue

            # separate_resnet / shared
            if f_i < 0:
                pose_pair = [inputs[("color_aug", f_i, 0)], inputs[("color_aug", 0, 0)]]
                invert_pose = True
            else:
                pose_pair = [inputs[("color_aug", 0, 0)], inputs[("color_aug", f_i, 0)]]
                invert_pose = False

            if self.opt.pose_model_type == "shared":
                # shared encoder: run the main encoder on each image
                feats0 = self.models["encoder"](pose_pair[0])
                feats1 = self.models["encoder"](pose_pair[1])
                deep0 = self._deep_feat(feats0)
                deep1 = self._deep_feat(feats1)
                axisangle, translation = self.models["pose"]([deep0, deep1])
            else:
                # separate_resnet: run pose_encoder on concatenated pair
                pair = torch.cat(pose_pair, 1)
                pose_feats = self.models["pose_encoder"](pair)
                deep = self._deep_feat(pose_feats)
                # IMPORTANT: PoseDecoder expects a list of feature tensors (one per input feature)
                axisangle, translation = self.models["pose"]([deep])

            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=invert_pose
            )

        return outputs

    def generate_images_pred(self, inputs, outputs):
        """Generate warped images for photometric reprojection loss."""
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            outputs[("depth", 0, scale)] = depth

            for frame_id in self.opt.frame_ids[1:]:
                if frame_id == "s":
                    continue

                T = outputs[("cam_T_cam", 0, frame_id)]
                cam_points = self.backproject_depth[scale](depth, inputs[("inv_K", scale)])
                pix_coords = self.project_3d[scale](cam_points, inputs[("K", scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords
                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, scale)],
                    pix_coords,
                    padding_mode="border",
                    align_corners=True
                )


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses."""
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        outputs = {}

        # depth prediction (target frame 0)
        features = self.models["encoder"](inputs[("color_aug", 0, 0)])
        outputs.update(self.models["depth"](features))

        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        # pose prediction (per-frame, Monodepth2-style)
        outputs.update(self.predict_poses(inputs))

        # warp images for photometric reprojection loss
        self.generate_images_pred(inputs, outputs)

        # Keep ASFMLearner extra heads (transform/position) computed (optional),
        # but make them safe by feeding PoseDecoder only the deepest feature map.
        try:
            transform_pair = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != 0], 1)
            transform_feats = self.models["transform_encoder"](transform_pair)
            deep_tr = self._deep_feat(transform_feats)
            axisangle_tr, translation_tr = self.models["transform"]([deep_tr])
            outputs.update({"axisangle_tr": axisangle_tr, "translation_tr": translation_tr})
        except Exception:
            pass

        try:
            position_pair = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != 0], 1)
            position_feats = self.models["position_encoder"](position_pair)
            deep_pos = self._deep_feat(position_feats)
            axisangle_pos, translation_pos = self.models["position"]([deep_pos])
            outputs.update({"axisangle_pos": axisangle_pos, "translation_pos": translation_pos})
        except Exception:
            pass

        losses = self.compute_losses(inputs, outputs)
        return outputs, losses

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if self.step % self.opt.log_frequency == 0:
                self.log("val", inputs, outputs, losses)

        self.set_train()

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                pred = inputs[("color", frame_id, scale)]
                identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, True)
                identity_reprojection_loss = identity_reprojection_losses.mean(1, True)
            else:
                reprojection_loss = reprojection_losses
                identity_reprojection_loss = identity_reprojection_losses

            if not self.opt.disable_automasking:
                identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), 1)
                reprojection_loss, idxs = torch.min(combined, dim=1)
                if self.opt.predictive_mask:
                    outputs["predictive_mask"] = torch.gather(outputs["predictive_mask"], 1, idxs.unsqueeze(1).unsqueeze(2).unsqueeze(3))
                    outputs["predictive_mask"] *= float(len(self.opt.frame_ids) - 1)

            if self.opt.predictive_mask:
                mask = outputs["predictive_mask"]
                reprojection_loss *= mask

            loss += reprojection_loss.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            return l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            return 0.85 * ssim_loss + 0.15 * l1_loss

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                    " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(
            self.epoch, batch_idx, samples_per_sec, loss,
            sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar(l, v, self.step)

        if self.wandb_enabled:
            wandb_losses = {f"{mode}/{k}": float(v.detach().cpu().item()) for k, v in losses.items()}
            wandb.log(wandb_losses, step=self.step)

        if mode == "train":
            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                writer.add_image(f"disp_{scale}", disp[0], self.step)
                if self.wandb_enabled and scale == 0:
                    disp_np = disp[0].detach().cpu().numpy()
                    disp_vis = cm.plasma((disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8))[:, :, :3]
                    wandb.log({f"{mode}/disp_scale0": wandb.Image(disp_vis)}, step=self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == "encoder":
                to_save["height"] = self.opt.height
                to_save["width"] = self.opt.width
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def print_params(self):
        """Print parameters for each model
        """
        print("Training model named:\n  ", self.opt.model_name)
        print("Models and parameters:")
        for k, v in self.models.items():
            num_params = sum([p.numel() for p in v.parameters()])
            print("  ", k, num_params)
        print("Training parameters:")
        print("  batch_size", self.opt.batch_size)
        print("  learning_rate", self.opt.learning_rate)
        print("  num_epochs", self.opt.num_epochs)
        print("  height", self.opt.height)
        print("  width", self.opt.width)
        print("  scales", self.opt.scales)
        print("  frame_ids", self.opt.frame_ids)
        print("  pose_model_type", self.opt.pose_model_type)
        print("  pose_model_input", self.opt.pose_model_input)
        print("  use_stereo", self.opt.use_stereo)
        print("  disable_automasking", self.opt.disable_automasking)
        print("  predictive_mask", self.opt.predictive_mask)
        print("  no_ssim", self.opt.no_ssim)
        print("  v1_multiscale", self.opt.v1_multiscale)
        print("  avg_reprojection", self.opt.avg_reprojection)
        print("  disparity_smoothness", self.opt.disparity_smoothness)
        print("  min_depth", self.opt.min_depth)
        print("  max_depth", self.opt.max_depth)
