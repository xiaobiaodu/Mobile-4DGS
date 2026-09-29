#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import os, random, time
from collections import deque
from dataclasses import dataclass
from lpipsPyTorch import lpips
from utils.loss_utils import image_gradient_loss, l1_loss
from fused_ssim import fused_ssim as fast_ssim
from gaussian_renderer import render_fluxgs, network_gui_ws
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.fast_utils import compute_gaussian_score_mobilegs2, sampling_cameras, sample_cameras_stratified, sample_cameras_temporal_stratified, compute_gaussian_pruning_mobilegs2
from utils.compress_utils import save_comp, write_storage, save_comp_web
from utils.resolution_scheduler import ResolutionSchedule
from utils.sh_utils import mc_project_sh_rgb, project_sh_mc
from torch import nn

# Module-level fallback only.  The effective dynamic batch size is taken from
# --mv (``pipe.mv``) at the start of ``training()``.
DYNAMIC_VIEW_BATCH_SIZE = 4


def prefetch_cameras(cameras, count, pop_from_end=False):
    count = min(max(int(count), 0), len(cameras))
    if count == 0:
        return
    if pop_from_end:
        candidates = reversed(cameras[-count:])
    else:
        candidates = cameras[:count]
    for camera in candidates:
        camera.prefetch()


def linear_ramp(iteration, start, end):
    """Return a stable 0->1 training weight over an inclusive schedule."""
    start = int(start)
    end = int(end)
    if end <= start:
        return 1.0 if iteration >= end else 0.0
    return min(max((iteration - start) / (end - start), 0.0), 1.0)


def draw_camera_batch(viewpoint_stack, all_cameras, batch_size, shuffle_fn=random.shuffle):
    """Draw one shuffled, without-replacement image batch.

    A short epoch tail is discarded before reshuffling, matching a DataLoader
    with ``drop_last=True``.  A sample here is a camera-frame pair, not merely a
    physical camera, so dynamic sequences retain uniform image sampling.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"view batch size must be positive, got {batch_size}")
    if len(all_cameras) < batch_size:
        raise ValueError(
            f"view batch size {batch_size} exceeds the {len(all_cameras)} "
            "available training images"
        )
    if len(viewpoint_stack) < batch_size:
        viewpoint_stack[:] = list(all_cameras)
        shuffle_fn(viewpoint_stack)
    return [viewpoint_stack.pop() for _ in range(batch_size)]


def create_camera_dataloader(cameras, batch_size):
    """Create the official-style shuffled camera-frame batch sampler.

    Camera instances already own CUDA transforms and lazy decoding state, so
    worker processes must not copy them.  The DataLoader is used for shuffled
    index sampling only; the existing thread prefetcher loads image pixels.
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"view batch size must be positive, got {batch_size}")
    if len(cameras) < batch_size:
        raise ValueError(
            f"view batch size {batch_size} exceeds the {len(cameras)} "
            "available training images"
        )
    return DataLoader(
        cameras,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list,
        drop_last=True,
    )


class CameraDataLoaderStream:
    """Infinite epoch stream over a shuffled, drop-last DataLoader.

    A small look-ahead queue preserves the repository's asynchronous frame
    prefetching without changing DataLoader's batch order.
    """

    def __init__(self, cameras, batch_size, prefetch_size=0):
        self.loader = create_camera_dataloader(cameras, batch_size)
        self._iterator = iter(self.loader)
        self._queue = deque()
        self._epoch_exhausted = False
        prefetch_size = max(int(prefetch_size), 0)
        self._prefetch_enabled = prefetch_size > 0
        requested_batches = max(
            1, (prefetch_size + int(batch_size) - 1) // int(batch_size)
        )
        self._queue_depth = min(requested_batches, len(self.loader))
        self._fill_queue()

    def __iter__(self):
        return self

    def __next__(self):
        if not self._queue:
            self._iterator = iter(self.loader)
            self._epoch_exhausted = False
            self._fill_queue()
        batch = self._queue.popleft()
        self._fill_queue()
        return batch

    def _next_loader_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._epoch_exhausted = True
            return None

    def _fill_queue(self):
        # Do not start the next epoch in the look-ahead queue. DataLoader's new
        # RandomSampler permutation is created only after every current-epoch
        # batch has actually trained, matching the normal ``for batch in
        # loader`` RNG timing.
        while len(self._queue) < self._queue_depth and not self._epoch_exhausted:
            batch = self._next_loader_batch()
            if batch is None:
                break
            if self._prefetch_enabled:
                prefetch_cameras(batch, len(batch))
            self._queue.append(batch)


@dataclass
class MultiviewDensificationStats:
    """Online aggregation of screen-space statistics for one optimizer batch."""

    xy_gradient_sum: torch.Tensor
    abs_gradient_sum: torch.Tensor
    visibility_count: torch.Tensor
    max_radii: torch.Tensor
    view_count: int = 0

    @classmethod
    def create(cls, point_count, device, dtype):
        point_count = int(point_count)
        column = torch.zeros((point_count, 1), device=device, dtype=dtype)
        return cls(
            xy_gradient_sum=column.clone(),
            abs_gradient_sum=column.clone(),
            visibility_count=column.clone(),
            max_radii=torch.zeros(point_count, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def add_view(self, viewspace_gradient, radii, point_indices=None):
        """Add gradients produced by a loss already divided by batch size."""
        if viewspace_gradient is None:
            raise RuntimeError("Rasterizer did not produce view-space gradients")
        if viewspace_gradient.ndim != 2 or viewspace_gradient.shape[1] < 3:
            raise ValueError(
                "view-space gradients must have shape [N, >=3], got "
                f"{tuple(viewspace_gradient.shape)}"
            )
        radii = radii.detach().reshape(-1)
        viewspace_gradient = viewspace_gradient.detach()
        local_count = viewspace_gradient.shape[0]
        if radii.shape[0] != local_count:
            raise ValueError(
                f"radii contain {radii.shape[0]} rows but gradients contain {local_count}"
            )
        full_view = point_indices is None
        if full_view:
            if local_count != self.max_radii.shape[0]:
                raise ValueError(
                    "point_indices are required when rendering a Gaussian subset"
                )
        else:
            point_indices = point_indices.detach().reshape(-1).long()
            if point_indices.shape[0] != local_count:
                raise ValueError(
                    "point_indices and view-space gradients must have the same row count"
                )

        visible_local = radii > 0
        xy_norm = torch.linalg.vector_norm(
            viewspace_gradient[visible_local, :2], dim=-1, keepdim=True
        )
        abs_norm = torch.linalg.vector_norm(
            viewspace_gradient[visible_local, 2:], dim=-1, keepdim=True
        )
        if full_view:
            self.xy_gradient_sum[visible_local] += xy_norm
            self.abs_gradient_sum[visible_local] += abs_norm
            self.visibility_count[visible_local] += 1
            self.max_radii[visible_local] = torch.maximum(
                self.max_radii[visible_local], radii[visible_local]
            )
        else:
            visible_points = point_indices[visible_local]
            self.xy_gradient_sum.index_add_(0, visible_points, xy_norm)
            self.abs_gradient_sum.index_add_(0, visible_points, abs_norm)
            self.visibility_count.index_add_(
                0, visible_points, torch.ones_like(xy_norm)
            )
            self.max_radii[visible_points] = torch.maximum(
                self.max_radii[visible_points], radii[visible_local]
            )
        self.view_count += 1

    @torch.no_grad()
    def finalize(self):
        """Return mean-visible unscaled gradients, visibility union and max radii."""
        if self.view_count <= 0:
            raise RuntimeError("cannot finalize an empty densification batch")
        visible = self.visibility_count[:, 0] > 0
        # Official 4D-GS treats the visible-view mean as one densification
        # observation per optimizer batch; GaussianModel therefore increments
        # denom once for this union rather than once per visible micro-view.
        correction = float(self.view_count) / self.visibility_count.clamp_min(1.0)
        return (
            self.xy_gradient_sum * correction,
            self.abs_gradient_sum * correction,
            visible,
            self.max_radii,
        )


def apply_attribute_stage_transition(gaussians, opt, iteration, net_itr, nn_iter):
    """Apply coincident attribute-stage changes with one optimizer rebuild."""
    start_net = iteration == net_itr
    start_sh_offset = iteration == nn_iter
    if not start_net and not start_sh_offset:
        return False

    # The projected static/view features are inputs to the attribute MLP. Build
    # them before enabling that MLP when both stages start at the same iteration.
    if start_sh_offset:
        static, view = project_sh_mc(
            gaussians.get_features,
            num_samples=opt.num_mc_points,
        )
        gaussians._features_static = nn.Parameter(static[:, 0].requires_grad_(True))
        gaussians._features_view = nn.Parameter(view[:, 0].requires_grad_(True))
        gaussians.active_sh_degree = 1
        gaussians.shoffset_enabled = True

    if start_net:
        gaussians.rot_feature_dim = opt.rot_feature_dim
        gaussians.construct_net()

    # This recreates the Gaussian optimizer and its parameter groups, so it must
    # run only after every transition scheduled for this iteration is applied.
    gaussians.training_setup(opt)

    if start_sh_offset:
        gaussians.init_shsnn(opt)

    return True


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, websockets, skip_quantize=False):
    global DYNAMIC_VIEW_BATCH_SIZE
    first_iter = 0
    # ModelParams and OptimizationParams are parsed separately.  Propagate the
    # requested mode before saving cfg_args and before Scene chooses its point
    # cloud so dynamic datasets can load their velocity initialization.
    dataset.dynamic = bool(opt.dynamic)
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    if gaussians.dynamic_enabled and not opt.dynamic:
        raise RuntimeError(
            "The input point cloud contains velocity initialization, but this is a static run. "
            "Pass --dynamic to use the initialized motion parameters."
        )
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    training_cameras = scene.getTrainCameras()
    configured_views = int(pipe.mv)
    # Both static and dynamic training use --mv (``pipe.mv``) as the number of
    # camera frames per optimizer update.  Dynamic sampling still shuffles the
    # camera-frame batches with drop_last=True.
    DYNAMIC_VIEW_BATCH_SIZE = configured_views
    batch_views = configured_views
    if batch_views <= 0:
        raise ValueError(f"--mv must be positive, got {batch_views}")
    if len(training_cameras) < batch_views:
        raise ValueError(
            f"--mv={batch_views} exceeds the {len(training_cameras)} training images"
        )
    has_lazy_frames = any(camera.image_path is not None for camera in training_cameras)
    if has_lazy_frames and opt.prefetch_size > 0 and opt.prefetch_workers > 0:
        Camera.configure_prefetch(opt.prefetch_workers)
        print(
            f"Frame prefetch enabled: {opt.prefetch_size} frames, "
            f"{opt.prefetch_workers} workers"
        )
    if opt.dynamic:
        dynamic_camera_batches = CameraDataLoaderStream(
            training_cameras,
            batch_size=batch_views,
            prefetch_size=opt.prefetch_size,
        )
        viewpoint_stack = None
        print(
            "Dynamic training sampler: DataLoader(shuffle=True, "
            f"batch_size={batch_views}, drop_last=True, num_workers=0)"
        )
    else:
        dynamic_camera_batches = None
        viewpoint_stack = training_cameras.copy()
        random.shuffle(viewpoint_stack)
        prefetch_cameras(viewpoint_stack, opt.prefetch_size, pop_from_end=True)
        print(
            f"Training with {batch_views} view(s) per optimizer update; "
            "camera-frame samples are shuffled without replacement"
        )

    # record time
    optim_start = torch.cuda.Event(enable_timing=True)
    optim_end = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    net_itr = getattr(opt, 'dynamic_net_itr', opt.net_itr) if opt.dynamic else opt.net_itr
    nn_iter = getattr(opt, 'dynamic_nn_iter', opt.nn_iter) if opt.dynamic else opt.nn_iter
    if opt.dynamic and net_itr != opt.net_itr:
        print(f"Dynamic scene: delaying attribute MLP stage to iteration {net_itr}")
    if opt.dynamic and nn_iter != opt.nn_iter:
        print(f"Dynamic scene: delaying SH-offset stage to iteration {nn_iter}")
    if opt.dynamic:
        print(
            "Dynamic temporal LR schedule: acceleration/time/duration decay "
            f"from iteration {opt.dynamic_temporal_lr_decay_start} to zero at "
            f"iteration {opt.dynamic_temporal_lr_freeze_iter}"
        )
    # The temporal codebooks are quantized one stage after the static ones, so the
    # 4D stage must leave fine-tuning room of its own.
    svq_4d_itr = getattr(opt, 'svq_4d_itr', opt.svq_itr)
    gate_freeze_iter = int(getattr(opt, 'dynamic_gate_freeze_iter', svq_4d_itr))
    if opt.dynamic and svq_4d_itr <= opt.svq_itr:
        raise ValueError(
            f"--svq_4d_itr ({svq_4d_itr}) must come after --svq_itr ({opt.svq_itr}); "
            "the 3D codebooks need a fine-tuning window before the 4D ones are built."
        )
    if opt.dynamic and not skip_quantize and gate_freeze_iter > svq_4d_itr:
        raise ValueError(
            f"--dynamic_gate_freeze_iter ({gate_freeze_iter}) must be no later than "
            f"--svq_4d_itr ({svq_4d_itr}); 4D SVQ needs a committed partition."
        )
    if not skip_quantize:
        for name, finetune_iters in (
            ("3D SVQ", (svq_4d_itr if opt.dynamic else opt.iterations) - opt.svq_itr),
            ("4D SVQ", opt.iterations - svq_4d_itr if opt.dynamic else None),
        ):
            if finetune_iters is not None and finetune_iters < 1000:
                print(
                    f"Warning: only {max(0, finetune_iters)} iterations after {name}; "
                    "quantized quality may be low. Lower --svq_itr/--svq_4d_itr or train longer."
                )
        print(
            f"Staged SVQ: 3D at iteration {opt.svq_itr}"
            + (f", 4D at iteration {svq_4d_itr}" if opt.dynamic else "")
            + f", encode at {opt.iterations}"
        )

    # DashGaussian-style multi-scale training.  The schedule is derived once,
    # before the first optimizer step, from the frequency content of a small
    # sample of training views; the loop below only queries it.  Densification
    # itself is untouched: only the training render and its target are resized.
    resolution_schedule = None
    if opt.multiscale:
        full_res_iter = int(opt.multiscale_full_res_iter)
        if full_res_iter <= 0:
            full_res_iter = int(opt.densify_until_iter)
        schedule_start = time.time()
        resolution_schedule = ResolutionSchedule.build(
            training_cameras,
            horizon=full_res_iter,
            max_scale=float(opt.multiscale_max_scale),
            num_levels=int(opt.multiscale_levels),
            num_views=int(opt.multiscale_num_images),
            search_max_scale=int(opt.multiscale_search_max_scale),
            seed=int(opt.multiscale_seed),
        )
        schedule_time = time.time() - schedule_start
        if resolution_schedule is None:
            print(
                "Multi-scale training requested but the derived schedule is flat "
                f"(--multiscale_max_scale {opt.multiscale_max_scale:g}); "
                f"training stays at full resolution ({schedule_time:.1f}s)"
            )
        else:
            print(f"{resolution_schedule.summary()}\n  (schedule derived in {schedule_time:.1f}s)")
    else:
        print("Multi-scale training disabled (--multiscale 0)")

    for iteration in range(first_iter, opt.iterations + 1):

        if websockets:
            if network_gui_ws.curr_id >= 0 and network_gui_ws.curr_id < len(scene.getTrainCameras()):
                cam = scene.getTrainCameras()[network_gui_ws.curr_id]
                net_image = render_fluxgs(cam, gaussians, pipe, background, opt.mult, 1.0)["render"]
                network_gui_ws.latest_width = cam.image_width
                network_gui_ws.latest_height = cam.image_height
                network_gui_ws.latest_result = net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

        iter_start.record()
        
        gaussians.update_learning_rate(iteration)

        if (
            opt.dynamic
            and iteration >= gate_freeze_iter
            and not gaussians.dynamic_gate_frozen
        ):
            dynamic_count = gaussians.commit_dynamic_partition()
            total_count = gaussians.get_xyz.shape[0]
            print(
                f"Committed strict dynamic partition at iteration {iteration}: "
                f"{dynamic_count:,}/{total_count:,} dynamic Gaussians"
            )

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        if iteration == opt.svq_itr and not skip_quantize:
            gaussians.apply_svq_3d(opt)

        if iteration == svq_4d_itr and not skip_quantize:
            if gaussians.dynamic_enabled and not gaussians.dynamic_gate_frozen:
                gaussians.commit_dynamic_partition()
            gaussians.apply_svq_4d(opt)


        # A view batch is a random group of camera-frame samples.  Draw the
        # whole batch before rendering so a short epoch tail is dropped rather
        # than mixed with the next shuffle.
        if dynamic_camera_batches is not None:
            viewpoint_batch = next(dynamic_camera_batches)
        else:
            viewpoint_batch = draw_camera_batch(
                viewpoint_stack, training_cameras, batch_views
            )
            for viewpoint_cam in viewpoint_batch:
                viewpoint_cam.prefetch()
            prefetch_cameras(
                viewpoint_stack, opt.prefetch_size, pop_from_end=True
            )
        if len(viewpoint_batch) != batch_views:
            raise RuntimeError(
                f"Expected a {batch_views}-view training batch, got "
                f"{len(viewpoint_batch)} samples"
            )

        collect_densification = iteration < opt.densify_until_iter
        batch_densification = (
            MultiviewDensificationStats.create(
                gaussians.get_xyz.shape[0],
                gaussians.get_xyz.device,
                gaussians.get_xyz.dtype,
            )
            if collect_densification else None
        )
        batch_l1_sum = gaussians.get_xyz.new_zeros(())
        batch_data_loss_sum = gaussians.get_xyz.new_zeros(())
        has_temporal_view = False

        # The gate is a scene-level partition.  During stochastic exploration,
        # share one detached noise realization across the four views, but
        # rebuild a separate ST graph for every sequential backward pass.
        if gaussians.dynamic_enabled:
            batch_gate_stochastic = gaussians.dynamic_gate_stochastic_active()
            batch_gate_noise = gaussians.sample_dynamic_gate_noise(
                stochastic=batch_gate_stochastic
            )
        else:
            batch_gate_stochastic = None
            batch_gate_noise = None

        # One resolution lookup per optimizer iteration, shared by every
        # micro-view of the batch so the whole batch is supervised at the same
        # scale.  The densification statistics collected below come from this
        # (possibly downsampled) render, exactly as before otherwise.
        render_scale = (
            resolution_schedule.scale_for(iteration)
            if resolution_schedule is not None else 1
        )
        if render_scale <= 1:
            render_scale = None

        for viewpoint_cam in viewpoint_batch:
            # Render one micro-view and immediately backpropagate its scaled
            # loss.  This is gradient accumulation with approximately
            # single-view peak graph memory, followed by one optimizer step.
            if (iteration - 1) == debug_from:
                pipe.debug = True

            # Progressive resolution: the supervised target and the rasterized
            # image share the resolution scheduled for this iteration.
            gt_image = viewpoint_cam.original_image.cuda()
            if render_scale is not None:
                # Antialiased downsampling keeps the low-resolution phase from
                # being supervised with aliased targets; the floored output
                # size is exactly what the rasterizer renders below.
                gt_image = torch.nn.functional.interpolate(gt_image[None], scale_factor=1/render_scale, mode="bilinear", 
                                                        recompute_scale_factor=True, antialias=True)[0]

            render_pkg = render_fluxgs(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                opt.mult,
                render_size=gt_image.shape[-2:],
                dynamic_gate_stochastic=batch_gate_stochastic,
                dynamic_gate_noise=batch_gate_noise,
            )
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            radii = render_pkg["radii"]

            # Loss
            view_l1 = l1_loss(image, gt_image)
            if iteration > net_itr and iteration <= net_itr + 100:
                view_loss = view_l1
            else:
                ssim_value = fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                view_loss = (
                    (1.0 - opt.lambda_dssim) * view_l1
                    + opt.lambda_dssim * (1.0 - ssim_value)
                )
            if opt.dynamic and getattr(opt, 'lambda_edge', 0.0) > 0:
                view_loss = view_loss + opt.lambda_edge * image_gradient_loss(
                    image, gt_image
                )
            if gaussians.dynamic_enabled and viewpoint_cam.time is not None:
                has_temporal_view = True

            (view_loss / batch_views).backward()
            batch_l1_sum += view_l1.detach()
            batch_data_loss_sum += view_loss.detach()

            if batch_densification is not None:
                batch_densification.add_view(
                    viewspace_point_tensor.grad,
                    radii,
                    (
                        None
                        if render_pkg.get("gaussian_subset") == "all"
                        else render_pkg.get("point_indices")
                    ),
                )

            # Do not keep the last full-resolution render graph alive while
            # processing the remaining views or the scene-level regularizers.
            del image, gt_image, view_l1, view_loss
            del viewspace_point_tensor, radii, render_pkg

        # Scene-level temporal and gate regularizers are independent of the
        # sampled camera.  Apply them exactly once per optimizer update rather
        # than once per micro-view.
        dynamic_regularization = gaussians.get_xyz.new_zeros(())
        gate_regularization = gaussians.get_xyz.new_zeros(())
        if gaussians.dynamic_enabled and has_temporal_view:
            with torch.no_grad():
                regularization_gate = gaussians.sample_dynamic_gate(
                    stochastic=batch_gate_stochastic,
                    logistic_noise=batch_gate_noise,
                )
            if getattr(opt, 'lambda_acceleration_reg', 0) > 0:
                dynamic_regularization = dynamic_regularization + (
                    opt.lambda_acceleration_reg
                    * gaussians.get_acceleration_regularization(
                        regularization_gate.detach()
                    )
                )
            if getattr(opt, 'lambda_duration_reg', 0) > 0:
                dynamic_regularization = dynamic_regularization + (
                    opt.lambda_duration_reg
                    * gaussians.get_duration_regularization(
                        regularization_gate,
                        getattr(opt, 'dynamic_duration_soft_max', 0.25),
                    )
                )
            if getattr(opt, 'lambda_motion_extent_reg', 0) > 0:
                dynamic_regularization = dynamic_regularization + (
                    opt.lambda_motion_extent_reg
                    * gaussians.get_motion_extent_regularization(
                        regularization_gate,
                        getattr(opt, 'dynamic_motion_extent_soft_max', 0.15),
                    )
                )

            sparsity_loss, binary_loss = gaussians.get_dynamic_gate_regularization()
            sparsity_ramp = linear_ramp(
                iteration,
                getattr(opt, 'dynamic_sparsity_start', 0),
                getattr(opt, 'dynamic_sparsity_ramp_end', 0),
            )
            binary_ramp = linear_ramp(
                iteration,
                getattr(opt, 'dynamic_binary_start', 0),
                gate_freeze_iter,
            )
            gate_regularization = (
                getattr(opt, 'lambda_dynamic_sparsity', 0.0)
                * sparsity_ramp
                * sparsity_loss
                + getattr(opt, 'lambda_gate_binary', 0.0)
                * binary_ramp
                * binary_loss
            )
            dynamic_regularization = dynamic_regularization + gate_regularization

        gate_regularization_gradient = None
        if (
            gate_regularization.requires_grad
            and isinstance(gaussians._dynamic_logit, torch.Tensor)
            and gaussians._dynamic_logit.requires_grad
        ):
            gate_regularization_gradient = torch.autograd.grad(
                gate_regularization,
                gaussians._dynamic_logit,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if dynamic_regularization.requires_grad:
            dynamic_regularization.backward()

        Ll1 = batch_l1_sum / batch_views
        loss = batch_data_loss_sum / batch_views + dynamic_regularization.detach()

        if batch_densification is not None:
            (
                batch_xy_gradient,
                batch_abs_gradient,
                batch_visibility_filter,
                batch_radii,
            ) = batch_densification.finalize()
        else:
            batch_xy_gradient = None
            batch_abs_gradient = None
            batch_visibility_filter = None
            batch_radii = None

        iter_end.record()


        # if iteration > 1000:
        #     print("dc grad is None:", gaussians._features_dc.grad is None)
        #     print("dc grad mean:", 
        #         None if gaussians._features_dc.grad is None 
        #         else gaussians._features_dc.grad.abs().mean().item())


        with torch.no_grad():
            promoted = gaussians.update_dynamic_promotions(
                iteration,
                opt,
                regularization_gradient=gate_regularization_gradient,
            )
            gaussians.mask_committed_static_gradients()
            if promoted:
                print(
                    f"Promoted {promoted:,} persistent reconstruction-error "
                    f"Gaussians to dynamic at iteration {iteration}"
                )
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Points": f"{gaussians.get_xyz.shape[0]:,}",
                    "Dynamic": (
                        f"{int(gaussians.get_dynamic_mask().sum().item()):,}"
                        if gaussians.dynamic_enabled else "0"
                    ),
                    "Res": "1/1" if render_scale is None else f"1/{render_scale}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            iter_time = iter_start.elapsed_time(iter_end)
            # Log and save
            # if iteration % 5000 == 0 and iteration >= 30000:
            #     print(len(gaussians.get_xyz))
            #     training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))
            
            optim_start.record()
            




            if iteration in testing_iterations:
                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_time, testing_iterations, scene, render_fluxgs, (pipe, background, opt.mult))

            if iteration == opt.iterations:
                if skip_quantize:
                    print("Skipping quantization (--skip_quantize)")
                else:
                    print("before quantization")
                    if not gaussians.vq_enabled:
                        if not gaussians.net_enabled:
                            raise RuntimeError("Quantization requires the neural attribute stage; increase --iterations/--dynamic_net_itr or use --skip_quantize.")
                        print("SVQ was not enabled during training; applying it now without fine-tuning.")
                        if gaussians.dynamic_enabled and not gaussians.dynamic_gate_frozen:
                            gaussians.commit_dynamic_partition()
                        gaussians.apply_svq(opt)
                    elif gaussians.dynamic_enabled and len(gaussians.dynamic_codes) == 0:
                        if not gaussians.dynamic_gate_frozen:
                            gaussians.commit_dynamic_partition()
                        gaussians.apply_svq_4d(opt)
                    save_dict = gaussians.encode()
                    save_comp_web(scene.model_path + "/comp.json", save_dict)

                    actual_storage = os.path.getsize(scene.model_path + "/comp.json")
                    with open(scene.model_path + "/storage.txt", 'w') as f:
                        byte = {'xyz': 0, 'scale':0, 'rotation':0, 'app':0, 'MLPs':0, 'opacity':0}
                        f.write(write_storage(save_dict, byte, gaussians.get_xyz.shape[0]))
                        f.write("Actual storage: " + str(round(actual_storage/2**20, 2)) + " MB")
                    # The live model already renders through its trained SVQ
                    # codebooks. Re-decoding the just-written artifact here is
                    # redundant and, after a long CUDA/cuML run, has repeatedly
                    # exposed corrupted Python decoder locals. Rendering below
                    # validates the artifact in a clean process instead.
                    print("compressed model saved; decode validation runs during rendering")


            # Densification
            if iteration < opt.densify_until_iter:
                # Every micro-view contributes.  Per-point screen gradients are
                # corrected back to their mean over visible views, visibility
                # is the union, and the radius is the batch-wise maximum.
                gaussians.max_radii2D[batch_visibility_filter] = torch.maximum(
                    gaussians.max_radii2D[batch_visibility_filter],
                    batch_radii[batch_visibility_filter],
                )
                gaussians.add_densification_stats_batch(
                    batch_xy_gradient,
                    batch_abs_gradient,
                    batch_visibility_filter,
                )

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    my_viewpoint_stack = scene.getTrainCameras().copy()
                    if gaussians.dynamic_enabled:
                        camlist = sample_cameras_temporal_stratified(
                            my_viewpoint_stack,
                            num_cams=opt.dynamic_score_cameras,
                            temporal_bins=opt.dynamic_score_time_bins,
                        )
                    else:
                        camlist = sample_cameras_stratified(my_viewpoint_stack)

                    prefetch_cameras(camlist, len(camlist))
                    importance_score, pruning_score = compute_gaussian_score_mobilegs2(camlist, gaussians, pipe, bg, opt, DENSIFY=True)                    
                    gaussians.densify_and_prune_mobilegs2(max_screen_size = size_threshold, 
                                                min_opacity = 0.005, 
                                                extent = scene.cameras_extent, 
                                                radii=batch_radii,
                                                args = opt,
                                                importance_score = importance_score,
                                                pruning_score = pruning_score,
                                                importance_quantile= opt.importance_quantile)

                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)) and iteration < net_itr:
                    gaussians.reset_opacity()

          
            if iteration % 3000 == 0 and iteration > 15_000 and iteration < 30_000:
                my_viewpoint_stack = scene.getTrainCameras().copy()
                if gaussians.dynamic_enabled:
                    camlist = sample_cameras_temporal_stratified(
                        my_viewpoint_stack,
                        num_cams=opt.dynamic_score_cameras,
                        temporal_bins=opt.dynamic_score_time_bins,
                    )
                else:
                    camlist = sample_cameras_stratified(my_viewpoint_stack)

                prefetch_cameras(camlist, len(camlist))
                pruning_score = compute_gaussian_pruning_mobilegs2(camlist, gaussians, pipe, bg, opt)                    
                gaussians.final_prune_mobilegs2(min_opacity = 0.1, pruning_score = pruning_score, pruning_quantile=opt.pruning_quantile)



            
            # Optimization step
            if iteration < opt.iterations:
                
                gaussians.optimizer.step()
                gaussians.constrain_dynamic_parameters(opt)
                gaussians.optimizer.zero_grad(set_to_none = True)
    
        
                if iteration >= opt.svq_itr and gaussians.vq_enabled:
                    gaussians.optimizer_code.step()
                    gaussians.constrain_dynamic_parameters(opt)
                    gaussians.optimizer_code.zero_grad(set_to_none=True)

                if iteration > nn_iter:
                    gaussians.shs_nn_optimizer.step()
                    gaussians.shs_nn_optimizer.zero_grad(set_to_none = True)

                if iteration > net_itr:
                    gaussians.optimizer_net.step()
                    gaussians.optimizer_net.zero_grad(set_to_none=True)
                    gaussians.scheduler_net.step()
            
            apply_attribute_stage_transition(
                gaussians,
                opt,
                iteration,
                net_itr,
                nn_iter,
            )


            # record time
            optim_end.record()
            torch.cuda.synchronize()
            optim_time = optim_start.elapsed_time(optim_end)
            total_time += (iter_time + optim_time) / 1e3

    scene.save(iteration)
    print(f"Gaussian number: {gaussians._xyz.shape[0]}")
    print(f"Training time: {total_time}")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    # if iteration in testing_iterations:
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                            {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test, ssim_test, lpips_test = 0.0, 0.0, 0.0
            for idx, viewpoint in enumerate(config['cameras']):
                image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                if tb_writer and (idx < 5):
                    tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    if iteration == testing_iterations[0]:
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
                ssim_test += fast_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()
                lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
            psnr_test /= len(config['cameras'])
            ssim_test /= len(config['cameras'])
            lpips_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])          
            print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
            if tb_writer:
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

    if tb_writer:
        tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--websockets", action='store_true', default=False)
    parser.add_argument("--benchmark_dir", type=str, default=None)
    parser.add_argument("--skip_quantize", action='store_true', default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if(args.websockets):
        network_gui_ws.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args.websockets,
        args.skip_quantize
    )

    # All done
    print("\nTraining complete.")
