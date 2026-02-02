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


_DEPTH_COLORMAP = cm.get_cmap('plasma', 256)  # for plotting


class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # Ensure input dimensions are multiples of 32 (required by encoder)
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []
        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        # Build models
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        # ASFMLearner extra heads
        self.models["transform_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["transform_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["transform_encoder"].parameters())

        self.models["transform"] = networks.PoseDecoder(
            self.models["transform_encoder"].num_ch_enc, num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["transform"].to(self.device)
        self.parameters_to_train += list(self.models["transform"].parameters())

        self.models["position_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained",
            num_input_images=self.num_pose_frames)
        self.models["position_encoder"].to(self.device)
        self.parameters_to_train += list(self.models["position_encoder"].parameters())

        self.models["position"] = networks.PoseDecoder(
            self.models["position_encoder"].num_ch_enc, num_input_features=1,
            num_frames_to_predict_for=2)
        self.models["position"].to(self.device)
        self.parameters_to_train += list(self.models["position"].parameters())

        # Pose network
        if self.use_pose_net:
            if self.opt.pose_model_type == "separate_resnet":
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers, self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)
                self.models["pose_encoder"].to(self.device)
                self.parameters_to_train += list(self.models["pose_encoder"].parameters())

                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc, num_input_features=1,
                    num_frames_to_predict_for=2)
            elif self.opt.pose_model_type == "shared":
                # IMPORTANT: shared pose uses TWO input feature tensors (one per image)
                self.models["pose"] = networks.PoseDecoder(
                    self.models["encoder"].num_ch_enc, num_input_features=2,
                    num_frames_to_predict_for=2)
            elif self.opt.pose_model_type == "posecnn":
                self.models["pose"] = networks.PoseCNN(
                    self.num_input_frames if self.opt.pose_model_input == "all" else 2)
            else:
                raise ValueError("Unknown pose_model_type: {}".format(self.opt.pose_model_type))

            self.models["pose"].to(self.device)
            self.parameters_to_train += list(self.models["pose"].parameters())

        if self.opt.predictive_mask:
            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales, num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and parameters:")
        for k, v in self.models.items():
            print("  ", k, sum(p.numel() for p in v.parameters()))
        print("Training parameters:")
        for k, v in vars(self.opt).items():
            print(" ", k, v)

        self.writer = SummaryWriter(self.log_path)

        if self.opt.use_wandb and wandb is not None:
            wandb.init(project=self.opt.wandb_project, name=self.opt.wandb_run_name, config=vars(self.opt))
            print("-> W&B enabled")
        else:
            print("-> W&B disabled")

        # Datasets
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "kitti_depth": datasets.KITTIDepthDataset,
            "hamlyn": datasets.HamlynDataset,
        }

        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))

        # If no val file exists, use test file as validation (common for Hamlyn)
        val_file = fpath.format("val")
        if os.path.exists(val_file):
            val_filenames = readlines(val_file)
        else:
            test_file = fpath.format("test")
            if os.path.exists(test_file):
                print(f"[WARN] No valid val_files.txt found for split '{self.opt.split}'. Using test_files.txt as validation.")
                val_filenames = readlines(test_file)
            else:
                print(f"[WARN] No val_files.txt or test_files.txt found for split '{self.opt.split}'. Validation will be empty.")
                val_filenames = []

        num_train_samples = len(train_filenames)
        num_val_samples = len(val_filenames)

        self.train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=self.opt.img_ext)

        self.val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=self.opt.img_ext)

        self.train_loader = DataLoader(
            self.train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            self.val_dataset, self.opt.batch_size, False,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)

        self.val_iter = iter(self.val_loader)

        self.epoch = 0
        self.step = 0

        # Prepare layers for view synthesis
        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"
        ]

        self.save_opts()

    # ----------------------------
    # Monodepth-style helpers
    # ----------------------------
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
                feats0 = self.models["encoder"](pose_pair[0])
                feats1 = self.models["encoder"](pose_pair[1])
                deep0 = self._deep_feat(feats0)
                deep1 = self._deep_feat(feats1)
                axisangle, translation = self.models["pose"]([deep0, deep1])
            else:
                pair = torch.cat(pose_pair, 1)
                pose_feats = self.models["pose_encoder"](pair)
                deep = self._deep_feat(pose_feats)
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

    def set_train(self):
        for m in self.models.values():
            m.train()

    def set_eval(self):
        for m in self.models.values():
            m.eval()

    def train(self):
        self.epoch = 0
        self.step = 0
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()

            # Save model checkpoints
            saved_this_epoch = False
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
                saved_this_epoch = True

            if bool(getattr(self.opt, "eval_each_epoch", False)):
                if not saved_this_epoch:
                    # We need a weights folder to evaluate against.
                    self.save_model()
                weights_folder = os.path.join(self.log_path, "models", f"weights_{self.epoch+1}")
                self.evaluate_each_epoch_if_enabled(weights_folder)

    def evaluate_each_epoch_if_enabled(self, weights_folder):
        """Runs depth evaluation using gt_depths.npz (e.g., Hamlyn) and logs metrics."""
        if not bool(getattr(self.opt, "eval_each_epoch", False)):
            return

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
            print(f"[WARN] Could not import evaluate_depth.py: {e}. Skipping evaluation.")
            return

        args = [
            "--eval_mono",
            "--dataset", self.opt.dataset,
            "--data_path", self.opt.data_path,
            "--load_weights_folder", weights_folder,
            "--eval_split", self.opt.eval_split,
            "--min_depth", str(self.opt.min_depth),
            "--max_depth", str(self.opt.max_depth),
        ]
        print("[EVAL] Running per-epoch evaluation:", " ".join(args))

        try:
            metrics = eval_depth.evaluate(args)
        except TypeError:
            # If evaluate signature differs, fallback to main-style call
            metrics = eval_depth.evaluate(args)

        if metrics is None:
            print("[EVAL] No metrics returned by evaluate_depth. Skipping logging.")
            return

        # metrics is expected to be a dict
        if self.opt.use_wandb and wandb is not None:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=self.step)
        for k, v in metrics.items():
            self.writer.add_scalar(f"eval/{k}", v, self.step)

    def run_epoch(self):
        self.model_lr_scheduler.step()

        self.set_train()

        print("Training")
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            if self.step % self.opt.log_frequency == 0:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_loss" in losses:
                    self.writer.add_scalar("depth_loss", losses["depth_loss"], self.step)

                self.log("train", inputs, outputs, losses)

            self.step += 1

            # GC to help OOM in long runs
            if self.step % 200 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            # occasional val logging
            if self.step % self.opt.log_frequency == 0:
                self.val()

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

        # ASFMLearner extra heads: keep them (safe deep feature feeding)
        transform_pair = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != 0], 1)
        transform_feats = self.models["transform_encoder"](transform_pair)
        deep_tr = self._deep_feat(transform_feats)
        axisangle_tr, translation_tr = self.models["transform"]([deep_tr])
        outputs.update({"axisangle_tr": axisangle_tr, "translation_tr": translation_tr})

        position_pair = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != 0], 1)
        position_feats = self.models["position_encoder"](position_pair)
        deep_pos = self._deep_feat(position_feats)
        axisangle_pos, translation_pos = self.models["position"]([deep_pos])
        outputs.update({"axisangle_pos": axisangle_pos, "translation_pos": translation_pos})

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

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, scale)]
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    identity_reprojection_loss = identity_reprojection_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if self.opt.predictive_mask:
                mask = outputs["predictive_mask"][("disp", scale)]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(mask, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)

                reprojection_loss *= mask

                loss += mask.mean()

            if not self.opt.disable_automasking:
                if self.opt.avg_reprojection:
                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)

                if combined.shape[1] == 2:
                    to_optimise, idxs = torch.min(combined, dim=1)
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if self.opt.predictive_mask:
                    outputs["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

                loss += to_optimise.mean()
            else:
                loss += reprojection_loss.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

            total_loss += loss
            losses["loss/{}".format(scale)] = loss

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        return losses

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def log_time(self, batch_idx, duration, loss):
        """Print a log statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0

        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
                       " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to tensorboard / wandb
        """
        writer = self.writer
        for l, v in losses.items():
            writer.add_scalar("{}/{}".format(mode, l), v, self.step)

        if self.opt.use_wandb and wandb is not None:
            wandb.log({f"{mode}/{k}": float(v) if hasattr(v, "item") else v for k, v in losses.items()},
                      step=self.step)

        for j in range(min(4, self.opt.batch_size)):
            for frame_id in self.opt.frame_ids:
                if frame_id == "s":
                    continue
                writer.add_image(
                    "{}/color_{}_{}/{}".format(mode, frame_id, 0, j),
                    inputs[("color", frame_id, 0)][j].data, self.step)

            for scale in self.opt.scales:
                disp = outputs[("disp", scale)]
                disp_resized = disp[j].data.cpu().numpy()
                disp_resized = disp_resized.squeeze()
                disp_resized = (_DEPTH_COLORMAP(disp_resized)[:, :, :3] * 255).astype(np.uint8)
                writer.add_image("{}/disp_{}/{}".format(mode, scale, j),
                                 disp_resized.transpose(2, 0, 1), self.step)

        # Optional wandb image logging
        if self.opt.use_wandb and wandb is not None and self.step % self.opt.log_frequency == 0:
            img_logs = {}
            try:
                # Log one example
                j = 0
                img_logs[f"{mode}/color_0"] = wandb.Image(inputs[("color", 0, 0)][j].permute(1, 2, 0).cpu().numpy())
                for frame_id in self.opt.frame_ids[1:]:
                    img_logs[f"{mode}/color_{frame_id}"] = wandb.Image(inputs[("color", frame_id, 0)][j].permute(1, 2, 0).cpu().numpy())
                for scale in self.opt.scales:
                    disp = outputs[("disp", scale)][j].detach().cpu().numpy().squeeze()
                    disp_img = (_DEPTH_COLORMAP(disp)[:, :, :3] * 255).astype(np.uint8)
                    img_logs[f"{mode}/disp_{scale}"] = wandb.Image(disp_img)
            except Exception:
                pass
            if img_logs:
                wandb.log(img_logs, step=self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        to_save = vars(self.opt).copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch + 1))
        os.makedirs(save_folder, exist_ok=True)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            torch.save(model.state_dict(), save_path)

        # save optimizer
        save_path = os.path.join(save_folder, "adam.pth")
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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights...")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
