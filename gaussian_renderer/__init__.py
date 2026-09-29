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
import math
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from diff_gaussian_rasterization_fluxgs import GaussianRasterizationSettings, GaussianRasterizer


GAUSSIAN_SUBSETS = ("all", "static", "dynamic")


def _resolve_gaussian_subset(pc, gaussian_subset):
    """Return original point indices for an exact static/dynamic render pass."""
    if gaussian_subset not in GAUSSIAN_SUBSETS:
        raise ValueError(
            f"gaussian_subset must be one of {GAUSSIAN_SUBSETS}, got "
            f"{gaussian_subset!r}"
        )

    point_count = pc.get_xyz.shape[0]
    device = pc.get_xyz.device
    if gaussian_subset == "all":
        return torch.arange(point_count, dtype=torch.long, device=device)

    if pc.dynamic_enabled:
        dynamic_mask = pc.get_dynamic_mask().reshape(-1)
        if dynamic_mask.shape[0] != point_count:
            raise ValueError(
                f"Dynamic mask has {dynamic_mask.shape[0]} entries for "
                f"{point_count} Gaussians"
            )
    else:
        dynamic_mask = torch.zeros(point_count, dtype=torch.bool, device=device)

    selected_mask = dynamic_mask if gaussian_subset == "dynamic" else ~dynamic_mask
    return torch.nonzero(selected_mask, as_tuple=False).reshape(-1)


def _select_point_rows(value, point_indices, point_count):
    if value is None or value.ndim == 0 or value.shape[0] != point_count:
        return value
    return value.index_select(0, point_indices)


def fixed_camera_cache_signature(
    viewpoint_camera,
    pipe,
    mult,
    scaling_modifier=1.0,
    render_size=None,
):
    """Return the immutable renderer state that makes a static cache valid."""
    height = (
        int(viewpoint_camera.image_height)
        if render_size is None else int(render_size[0])
    )
    width = (
        int(viewpoint_camera.image_width)
        if render_size is None else int(render_size[1])
    )

    def array_bytes(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous().numpy().tobytes()
        try:
            import numpy as np
            return np.asarray(value).tobytes()
        except Exception:
            return repr(value)

    rotation = getattr(viewpoint_camera, "R", None)
    translation = getattr(viewpoint_camera, "T", None)
    fallback_view = (
        getattr(viewpoint_camera, "world_view_transform", None)
        if rotation is None or translation is None else None
    )
    return (
        array_bytes(rotation),
        array_bytes(translation),
        array_bytes(fallback_view),
        array_bytes(getattr(viewpoint_camera, "trans", None)),
        float(getattr(viewpoint_camera, "scale", 1.0)),
        float(viewpoint_camera.FoVx),
        float(viewpoint_camera.FoVy),
        float(getattr(viewpoint_camera, "znear", 0.01)),
        float(getattr(viewpoint_camera, "zfar", 100.0)),
        height,
        width,
        float(mult),
        float(scaling_modifier),
        bool(pipe.compute_cov3D_python),
        bool(pipe.convert_SHs_python),
        bool(pipe.debug),
    )


def _materialized_subset_inputs(
    viewpoint_camera,
    pc,
    pipe,
    point_indices,
    scaling_modifier,
    dynamic,
):
    """Prepare only the compact branch used by inference-time caching."""
    if pc.net_enabled or pc.vq_enabled or pc.shoffset_enabled:
        raise RuntimeError(
            "Exact static caching currently requires materialized Gaussian "
            "attributes (PLY or --decode output)"
        )

    xyz = pc._xyz.index_select(0, point_indices)
    if dynamic:
        if not pc.dynamic_enabled:
            raise RuntimeError("A dynamic cache branch requires a dynamic model")
        velocity = pc._velocity.index_select(0, point_indices)
        if pc._acceleration.numel() == pc._velocity.numel():
            acceleration = pc._acceleration.index_select(0, point_indices)
        else:
            acceleration = torch.zeros_like(velocity)
        canonical_time = pc._time.index_select(0, point_indices)
        raw_duration = pc._duration.index_select(0, point_indices)
        frame_time = torch.as_tensor(
            viewpoint_camera.time, dtype=xyz.dtype, device=xyz.device
        ).reshape(1, 1)
        dt = frame_time - canonical_time
        # Match GaussianModel.get_deformed_xyz exactly, including floating
        # point association.  Reassociating this as
        # ``(xyz + velocity * dt) + ...`` can move a few primitives across a
        # tile/depth/alpha boundary even though the real-valued formula is the
        # same.
        offset = velocity * dt + 0.5 * acceleration * dt.square()
        means3D = xyz + offset
        duration = torch.exp(raw_duration)
        if pc.dynamic_duration_max is None:
            duration = duration.clamp(min=pc.dynamic_duration_min)
        else:
            duration = duration.clamp(
                min=pc.dynamic_duration_min,
                max=pc.dynamic_duration_max,
            )
        temporal_opacity = torch.exp(
            -0.5 * ((frame_time - canonical_time) / (duration + 1e-8)) ** 2
        )
    else:
        means3D = xyz
        temporal_opacity = torch.ones(
            (point_indices.numel(), 1), dtype=xyz.dtype, device=xyz.device
        )

    base_opacity = pc.opacity_activation(
        pc._opacity.index_select(0, point_indices)
    )
    opacity = base_opacity * temporal_opacity
    scaling = pc.scaling_activation(pc._scaling.index_select(0, point_indices))
    raw_rotation = pc._rotation.index_select(0, point_indices)
    rotation = pc.rotation_activation(raw_rotation)
    shs = torch.cat(
        (
            pc._features_dc.index_select(0, point_indices),
            pc._features_rest.index_select(0, point_indices),
        ),
        dim=1,
    )

    colors_precomp = None
    if pipe.convert_SHs_python:
        shs_view = shs.transpose(1, 2).reshape(
            -1, 3, (pc.max_sh_degree + 1) ** 2
        )
        direction = means3D - viewpoint_camera.camera_center.reshape(1, 3)
        direction = direction / direction.norm(dim=1, keepdim=True)
        colors_precomp = torch.clamp_min(
            eval_sh(pc.active_sh_degree, shs_view, direction) + 0.5,
            0.0,
        )
        shs = None

    cov3D_precomp = None
    scales = scaling
    rotations = rotation
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.covariance_activation(
            scaling,
            scaling_modifier,
            raw_rotation,
        )
        scales = None
        rotations = None

    return {
        "means3D": means3D.contiguous(),
        "opacities": opacity.contiguous(),
        "shs": None if shs is None else shs.contiguous(),
        "colors_precomp": (
            None if colors_precomp is None else colors_precomp.contiguous()
        ),
        "scales": None if scales is None else scales.contiguous(),
        "rotations": None if rotations is None else rotations.contiguous(),
        "cov3D_precomp": (
            None if cov3D_precomp is None else cov3D_precomp.contiguous()
        ),
        "base_opacity": base_opacity,
        "temporal_opacity": temporal_opacity,
    }


class ExactStaticCacheRenderer:
    """Exact fixed-camera inference renderer with one cached static stream."""

    def __init__(
        self,
        pc,
        pipe,
        bg_color,
        mult,
        scaling_modifier=1.0,
        render_size=None,
    ):
        if torch.is_grad_enabled():
            raise RuntimeError("ExactStaticCacheRenderer must be created in torch.no_grad()")
        if not pc.dynamic_enabled:
            raise ValueError("Exact static caching requires a dynamic Gaussian model")
        if not pc.dynamic_gate_frozen:
            raise RuntimeError("Commit/freeze the dynamic partition before caching")
        if pc.net_enabled or pc.vq_enabled or pc.shoffset_enabled:
            raise RuntimeError(
                "Exact static caching requires materialized attributes; load the "
                "PLY checkpoint or render the compressed artifact with --decode"
            )

        self.pc = pc
        self.pipe = pipe
        self.bg_color = bg_color
        self.mult = float(mult)
        self.scaling_modifier = float(scaling_modifier)
        self.render_size = render_size
        dynamic_mask = pc.get_dynamic_mask().reshape(-1)
        self.static_indices = torch.nonzero(
            ~dynamic_mask, as_tuple=False
        ).reshape(-1).contiguous()
        self.dynamic_indices = torch.nonzero(
            dynamic_mask, as_tuple=False
        ).reshape(-1).contiguous()
        self.point_count = int(dynamic_mask.numel())
        self.signature = None
        self.cache = None
        self.metric_map = None
        self.build_ms = None
        self.hits = 0

    def _rasterizer(self, viewpoint_camera):
        height = (
            int(viewpoint_camera.image_height)
            if self.render_size is None else int(self.render_size[0])
        )
        width = (
            int(viewpoint_camera.image_width)
            if self.render_size is None else int(self.render_size[1])
        )
        if self.metric_map is None or self.metric_map.numel() != height * width:
            self.metric_map = torch.zeros(
                height * width, dtype=torch.int32, device=self.bg_color.device
            )
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=math.tan(viewpoint_camera.FoVx * 0.5),
            tanfovy=math.tan(viewpoint_camera.FoVy * 0.5),
            bg=self.bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            mult=self.mult,
            prefiltered=False,
            debug=self.pipe.debug,
            get_flag=False,
            metric_map=self.metric_map,
        )
        rasterizer = GaussianRasterizer(raster_settings=settings)
        if not hasattr(rasterizer, "build_static_cache"):
            raise RuntimeError(
                "The installed FluxGS extension wrapper has no static-cache API. "
                "Reinstall submodules/diff-gaussian-rasterization_fluxgs."
            )
        return rasterizer

    def _validate_camera(self, viewpoint_camera):
        signature = fixed_camera_cache_signature(
            viewpoint_camera,
            self.pipe,
            self.mult,
            self.scaling_modifier,
            self.render_size,
        )
        if self.signature is not None and signature != self.signature:
            raise RuntimeError(
                "Exact static cache camera changed; this renderer intentionally "
                "uses one fixed-camera cache"
            )
        return signature

    def build(self, viewpoint_camera):
        if self.cache is not None:
            return self.cache
        signature = self._validate_camera(viewpoint_camera)
        rasterizer = self._rasterizer(viewpoint_camera)
        static_inputs = _materialized_subset_inputs(
            viewpoint_camera,
            self.pc,
            self.pipe,
            self.static_indices,
            self.scaling_modifier,
            dynamic=False,
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self.cache = rasterizer.build_static_cache(
            means3D=static_inputs["means3D"],
            opacities=static_inputs["opacities"],
            global_ids=self.static_indices,
            total_point_count=self.point_count,
            shs=static_inputs["shs"],
            colors_precomp=static_inputs["colors_precomp"],
            scales=static_inputs["scales"],
            rotations=static_inputs["rotations"],
            cov3D_precomp=static_inputs["cov3D_precomp"],
        )
        end.record()
        end.synchronize()
        self.build_ms = float(start.elapsed_time(end))
        self.signature = signature
        return self.cache

    def render(self, viewpoint_camera):
        self._validate_camera(viewpoint_camera)
        if self.cache is None:
            self.build(viewpoint_camera)
        rasterizer = self._rasterizer(viewpoint_camera)
        dynamic_inputs = _materialized_subset_inputs(
            viewpoint_camera,
            self.pc,
            self.pipe,
            self.dynamic_indices,
            self.scaling_modifier,
            dynamic=True,
        )
        rendered_image, dynamic_radii = rasterizer.forward_cached(
            self.cache,
            means3D=dynamic_inputs["means3D"],
            opacities=dynamic_inputs["opacities"],
            global_ids=self.dynamic_indices,
            shs=dynamic_inputs["shs"],
            colors_precomp=dynamic_inputs["colors_precomp"],
            scales=dynamic_inputs["scales"],
            rotations=dynamic_inputs["rotations"],
            cov3D_precomp=dynamic_inputs["cov3D_precomp"],
        )
        self.hits += 1
        return {
            "render": rendered_image,
            "radii": dynamic_radii,
            "point_indices": self.dynamic_indices,
            "gaussian_subset": "all",
            "base_opacity": dynamic_inputs["base_opacity"],
            "temporal_opacity": dynamic_inputs["temporal_opacity"],
            "cache_hit": True,
        }


def render_fluxgs(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, mult, scaling_modifier = 1.0, override_color = None, get_flag=None, metric_map = None, render_size=None, gaussian_subset="all", dynamic_gate_stochastic=None, dynamic_gate_noise=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    point_count = pc.get_xyz.shape[0]
    point_indices = _resolve_gaussian_subset(pc, gaussian_subset)
    render_all = gaussian_subset == "all"

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points = torch.zeros((point_indices.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    render_height = int(viewpoint_camera.image_height) if render_size is None else int(render_size[0])
    render_width = int(viewpoint_camera.image_width) if render_size is None else int(render_size[1])

    if metric_map==None:
        # The rasterizer indexes metric_map as W * y + x over the image it is
        # actually rendering, so the default buffer has to follow render_size
        # rather than the camera resolution (they differ while training with a
        # downsampled multi-scale render target).
        metric_map=torch.zeros(render_height * render_width, dtype=torch.int32, device='cuda')

    raster_settings = GaussianRasterizationSettings(
        image_height=render_height,
        image_width=render_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        mult = mult,
        prefiltered=False,
        debug=pipe.debug,
        get_flag=get_flag,
        metric_map = metric_map
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    cam_time = getattr(viewpoint_camera, 'time', None)
    dynamic_gate = (
        pc.sample_dynamic_gate(
            stochastic=dynamic_gate_stochastic,
            logistic_noise=dynamic_gate_noise,
        )
        if pc.dynamic_enabled else None
    )
    all_means3D = pc.get_deformed_xyz(cam_time, gate=dynamic_gate) if pc.dynamic_enabled else pc.get_xyz
    means3D = all_means3D if render_all else all_means3D.index_select(0, point_indices)
    means2D = screenspace_points

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        if pc.vq_enabled:
            cov3D_precomp = pc.covariance_activation(pc.get_svq_scale, scaling_modifier, pc.get_svq_rotation)
        else:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        opacity = pc.get_opacity
    elif pc.vq_enabled:
        scales = pc.get_svq_scale
        rotations = pc.get_svq_rotation
        # opacity = pc.get_svq_opacity
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        opacity = pc.get_opacity

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (all_means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            
            if pc.net_enabled:
                cont_feature = pc.mlp_cont(pc._contract_xyz(pc.get_xyz.detach()))
                if pc.vq_enabled:
                    app_feature = pc.get_svq_appearance
                    space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
                    view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)
                else:
                    space_feature = torch.cat([cont_feature, pc._features_static],dim=-1)
                    view_feature = torch.cat([cont_feature, pc._features_view],dim=-1)
                shs = pc.mlp_view(view_feature).reshape(-1,pc.max_sh_rest,3).float()
                dc = pc.mlp_dc(space_feature).reshape(-1,1,3).float()
                opacity = pc.opacity_activation(pc.mlp_opacity(space_feature).float())
                shs = torch.cat([dc, shs], dim=1)

                if pc.vq_enabled:
                    rot_feature = pc.get_svq_rot_feature
                else:
                    rot_feature = pc._features_rot
                rot_input = torch.cat([cont_feature, rot_feature], dim=-1)
                rotations = torch.nn.functional.normalize(pc.mlp_rotation(rot_input).float(), dim=-1)

                shs = shs + pc.get_features_offset(shs, opacity, rotations, all_means3D)

            elif pc.shoffset_enabled:
                shs = pc.get_features 
                shs = shs + pc.get_features_offset(shs, opacity, xyz=all_means3D)
            else:
                shs = pc.get_features 

    else:
        colors_precomp = override_color

    base_opacity = opacity
    temporal_opacity = None
    if pc.dynamic_enabled:
        temporal_opacity = pc.get_temporal_opacity(cam_time, gate=dynamic_gate)
        opacity = opacity * temporal_opacity

    if not render_all:
        base_opacity = _select_point_rows(base_opacity, point_indices, point_count)
        temporal_opacity = _select_point_rows(temporal_opacity, point_indices, point_count)
        opacity = _select_point_rows(opacity, point_indices, point_count)
        scales = _select_point_rows(scales, point_indices, point_count)
        rotations = _select_point_rows(rotations, point_indices, point_count)
        cov3D_precomp = _select_point_rows(cov3D_precomp, point_indices, point_count)
        shs = _select_point_rows(shs, point_indices, point_count)
        colors_precomp = _select_point_rows(colors_precomp, point_indices, point_count)

    if point_indices.numel() == 0:
        image_height = int(viewpoint_camera.image_height) if render_size is None else render_size[0]
        image_width = int(viewpoint_camera.image_width) if render_size is None else render_size[1]
        rendered_image = bg_color[:, None, None].expand(
            bg_color.shape[0], image_height, image_width
        ).clone()
        radii = torch.empty(0, dtype=pc.get_xyz.dtype, device=pc.get_xyz.device)
        accum_metric_counts = torch.zeros(
            image_height * image_width, dtype=torch.float32, device=pc.get_xyz.device
        )
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": torch.empty(
                (0, 1), dtype=torch.long, device=pc.get_xyz.device
            ),
            "radii": radii,
            "base_opacity": base_opacity,
            "temporal_opacity": temporal_opacity,
            "dynamic_gate": dynamic_gate,
            "point_indices": point_indices,
            "gaussian_subset": gaussian_subset,
            "accum_metric_counts": accum_metric_counts,
        }


  
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, accum_metric_counts = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : (radii > 0).nonzero(),
            "radii": radii,
            "base_opacity": base_opacity,
            "temporal_opacity": temporal_opacity,
            "dynamic_gate": dynamic_gate,
            "point_indices": point_indices,
            "gaussian_subset": gaussian_subset,
            "accum_metric_counts" : accum_metric_counts}
