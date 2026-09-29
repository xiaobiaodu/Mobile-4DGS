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

from dataclasses import dataclass
from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


@dataclass(frozen=True)
class StaticGaussianRasterCache:
    """Inference-only projected/sorted data for one fixed camera.

    The three ``combined_*`` tensors are persistent compact storage.  Their
    static prefix is written once when the cache is built and their dynamic
    suffix is overwritten by each cached frame.
    """

    point_count: int
    static_point_count: int
    static_rendered_count: int
    static_entries: torch.Tensor
    combined_means2D: torch.Tensor
    combined_conic_opacity: torch.Tensor
    combined_features: torch.Tensor

    @property
    def resident_bytes(self):
        tensors = (
            self.static_entries,
            self.combined_means2D,
            self.combined_conic_opacity,
            self.combined_features,
        )
        return sum(t.numel() * t.element_size() for t in tensors)


def _extension_args(
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    get_flag = raster_settings.get_flag
    if get_flag is None:
        get_flag = False

    # The standard Camera stores 4x4 transforms as transposed tensor views.
    # The original rasterizer materializes every input with contiguous() in
    # C++; do the same at this public cache boundary because cached forward
    # intentionally validates and reads the tensors directly.  In particular,
    # passing a stride-[1, 4] matrix's raw data pointer would change its
    # physical glm layout rather than merely being a performance issue.
    def contiguous(tensor):
        return tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor

    return (
        contiguous(raster_settings.bg),
        contiguous(means3D),
        contiguous(colors_precomp),
        contiguous(opacities),
        contiguous(scales),
        contiguous(rotations),
        raster_settings.scale_modifier,
        contiguous(cov3Ds_precomp),
        contiguous(raster_settings.metric_map),
        contiguous(raster_settings.viewmatrix),
        contiguous(raster_settings.projmatrix),
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        contiguous(sh),
        raster_settings.sh_degree,
        contiguous(raster_settings.campos),
        raster_settings.mult,
        raster_settings.prefiltered,
        raster_settings.debug,
        get_flag,
    )


def build_static_gaussian_cache(
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    global_ids,
    total_point_count,
    raster_settings,
):
    """Project and sort the static subset once for fixed-camera inference."""
    if torch.is_grad_enabled():
        raise RuntimeError("Static Gaussian caching is inference-only; use torch.no_grad()")
    if bool(raster_settings.get_flag):
        raise ValueError("Static Gaussian caching does not support get_flag/metric accumulation")
    if not hasattr(_C, "build_static_cache"):
        raise RuntimeError(
            "The FluxGS rasterizer extension has no static-cache support. "
            "Reinstall submodules/diff-gaussian-rasterization_fluxgs."
        )

    args = _extension_args(
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    result = _C.rasterize_gaussians(*args)
    num_rendered = int(result[0])
    geom_buffer = result[4]
    binning_buffer = result[5]
    (
        static_entries,
        combined_means2D,
        combined_conic_opacity,
        combined_features,
    ) = _C.build_static_cache(
        geom_buffer,
        binning_buffer,
        colors_precomp.contiguous(),
        global_ids.contiguous(),
        int(means3D.shape[0]),
        num_rendered,
        int(total_point_count),
    )
    return StaticGaussianRasterCache(
        point_count=int(total_point_count),
        static_point_count=int(means3D.shape[0]),
        static_rendered_count=num_rendered,
        static_entries=static_entries,
        combined_means2D=combined_means2D,
        combined_conic_opacity=combined_conic_opacity,
        combined_features=combined_features,
    )


def rasterize_gaussians_with_static_cache(
    cache,
    means3D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    global_ids,
    raster_settings,
):
    """Render one frame by sorting only dynamic Gaussians and exactly merging."""
    if torch.is_grad_enabled():
        raise RuntimeError("Static Gaussian caching is inference-only; use torch.no_grad()")
    if bool(raster_settings.get_flag):
        raise ValueError("Static Gaussian caching does not support get_flag/metric accumulation")
    if not hasattr(_C, "rasterize_gaussians_cached"):
        raise RuntimeError(
            "The FluxGS rasterizer extension has no static-cache support. "
            "Reinstall submodules/diff-gaussian-rasterization_fluxgs."
        )
    if cache.static_point_count + int(means3D.shape[0]) != cache.point_count:
        raise ValueError(
            "Cached static and current dynamic point counts do not reconstruct "
            f"the model: {cache.static_point_count} + {means3D.shape[0]} != "
            f"{cache.point_count}"
        )

    args = _extension_args(
        means3D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )
    result = _C.rasterize_gaussians_cached(
        *args,
        cache.static_entries,
        cache.combined_means2D,
        cache.combined_conic_opacity,
        cache.combined_features,
        cache.static_point_count,
        global_ids.contiguous(),
    )
    return result[0], result[1]

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings
    ):

        # Restructure arguments the way that the C++ lib expects them
        get_flag = raster_settings.get_flag
        if get_flag == None:
            get_flag = False

        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.metric_map,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.mult,
            raster_settings.prefiltered,
            raster_settings.debug,
            get_flag
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, num_buckets, color, radii, geomBuffer, binningBuffer, imgBuffer, sampleBuffer, accum_metric_counts = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.num_buckets = num_buckets
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer)
        return color, radii, accum_metric_counts

    @staticmethod
    def backward(ctx, grad_out_color, _, g_metric):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        num_buckets = ctx.num_buckets
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, sampleBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                num_buckets,
                sampleBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    mult : float
    prefiltered : bool
    debug : bool
    get_flag : bool
    metric_map : torch.Tensor

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
  
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings
        )

    def build_static_cache(
        self,
        means3D,
        opacities,
        global_ids,
        total_point_count,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Build one immutable-camera static cache without an autograd graph."""
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise ValueError("Provide exactly one of shs or colors_precomp")
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise ValueError("Provide exactly one of scale/rotation or cov3D_precomp")

        empty = means3D.new_empty(0)
        return build_static_gaussian_cache(
            means3D,
            empty if shs is None else shs,
            empty if colors_precomp is None else colors_precomp,
            opacities,
            empty if scales is None else scales,
            empty if rotations is None else rotations,
            empty if cov3D_precomp is None else cov3D_precomp,
            global_ids,
            total_point_count,
            self.raster_settings,
        )

    def forward_cached(
        self,
        cache,
        means3D,
        opacities,
        global_ids,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None,
    ):
        """Render dynamic rows against an exact fixed-camera static cache."""
        if (shs is None and colors_precomp is None) or (
            shs is not None and colors_precomp is not None
        ):
            raise ValueError("Provide exactly one of shs or colors_precomp")
        if ((scales is None or rotations is None) and cov3D_precomp is None) or (
            (scales is not None or rotations is not None) and cov3D_precomp is not None
        ):
            raise ValueError("Provide exactly one of scale/rotation or cov3D_precomp")

        empty = means3D.new_empty(0)
        return rasterize_gaussians_with_static_cache(
            cache,
            means3D,
            empty if shs is None else shs,
            empty if colors_precomp is None else colors_precomp,
            opacities,
            empty if scales is None else scales,
            empty if rotations is None else rotations,
            empty if cov3D_precomp is None else cov3D_precomp,
            global_ids,
            self.raster_settings,
        )

class SparseGaussianAdam(torch.optim.Adam):
    def __init__(self, params, lr, eps):
        super().__init__(params=params, lr=lr, eps=eps)
    
    @torch.no_grad()
    def step(self, visibility, N):
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]

            assert len(group["params"]) == 1, "more than one tensor in group"
            param = group["params"][0]
            if param.grad is None:
                continue

            # Lazy state initialization
            state = self.state[param]
            if len(state) == 0:
                state['step'] = torch.tensor(0.0, dtype=torch.float32)
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)


            stored_state = self.state.get(param, None)
            exp_avg = stored_state["exp_avg"]
            exp_avg_sq = stored_state["exp_avg_sq"]
            M = param.numel() // N
            _C.adamUpdate(param, param.grad, exp_avg, exp_avg_sq, visibility, lr, 0.9, 0.999, eps, N, M)
