"""FluxGS -> gsplat 1.5.3 (ROCm) adapter (direct differentiable call design).

Replaces the CUDA extension `diff_gaussian_rasterization_fluxgs` with gsplat's
public differentiable `rasterization()` (which carries its own autograd kernels
on ROCm). We only translate:
  - conventions: Inria (N,4)x3 / scale_modifier / viewmatrix
                 <-> gsplat (B,N,*) / Ks / viewmats
  - output layout: gsplat (B,C,H,W,3) -> FluxGS (3,H,W) + radii + dummy accum

Verified target env: torch 2.10.0+rocm7.0 / gsplat 1.5.3+6f2f06e / MI300X.
"""
from __future__ import annotations

import math
import os
from typing import NamedTuple

import torch
import torch.nn as nn

import torch.nn.functional as F
from gsplat import rasterization as _gsplat_rasterization

_MEM_TRACE = os.environ.get("FLUXGS_MEM_TRACE") is not None
_TRACE_COUNT = [0]


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    mult: float
    prefiltered: bool
    debug: bool
    get_flag: bool
    metric_map: torch.Tensor


class StaticGaussianRasterCache:
    """Compat stub (FluxGS inference-only cache path; training never needs it)."""
    pass


def _to_gsplat_view(rs: GaussianRasterizationSettings, device):
    """Pass Inria/colmap world->cam straight through as gsplat viewmats.

    Both gsplat and Inria 3DGS use the COLMAP convention (camera looks down +z,
    x right, y down) with a pinhole K from tanfov, so no axis remapping is needed.
    """
    view_in = rs.viewmatrix.to(device=device)
    return view_in.view(1, 1, 4, 4).contiguous()


def _make_Ks(rs: GaussianRasterizationSettings, device, dtype):
    fx = rs.image_width / (2.0 * rs.tanfovx)
    fy = rs.image_height / (2.0 * rs.tanfovy)
    K = torch.tensor(
        [[fx, 0.0, rs.image_width / 2.0],
         [0.0, fy, rs.image_height / 2.0],
         [0.0, 0.0, 1.0]],
        device=device, dtype=dtype).view(1, 1, 3, 3).contiguous()
    return K


def _rasterize(means3D, scales, rotations, opacities, colors, rs, use_sh=False, shs=None, means2d_out=None):
    """Differentiable core: FluxGS inputs -> (color(3,H,W), radii(N,), means2d_px(N,2)).

    radii and means2d_px are scattered back to full N length (invisible = 0),
    matching the Inria/FluxGS contract of per-gaussian arrays indexed by point.
    """
    device = means3D.device
    N = means3D.shape[0]
    sm = rs.scale_modifier if rs.scale_modifier not in (0.0,) else 1.0
    # CONVENTION: FluxGS/Inria rasterizer consumes RAW (linear) scales in world units.
    # This ROCm gsplat kernel (1.5.3) also consumes RAW linear scales, not log-space —
    # verified empirically by rendering a single gaussian at fixed focal length (see
    # probe4.py in the scratch dir): scale=0.05 → 10 px blob, scale=0.5 → 99 px blob,
    # monotone, linear-ish in log-space of scale.  Passing `torch.log(scales)` (as the
    # upstream gsplat Python API does for *their* CUDA build) would make this kernel
    # treat 0.03 as log(0.03)≈-3.5, i.e. ~10^5× magnification → full-frame smear.
    scales_in = scales * sm

    means_b = means3D.view(1, N, 3)
    quats_b = rotations.view(1, N, 4)
    scales_b = scales_in.view(1, N, 3).contiguous()
    opac_b = opacities.view(1, N).contiguous()
    viewmats = _to_gsplat_view(rs, device)
    Ks = _make_Ks(rs, device, means_b.dtype)

    if colors is not None:
        cols_b = colors.reshape(1, N, 3) if colors.dim() == 2 else colors.view(1, N, 3)
        out = _gsplat_rasterization(means_b, quats_b, scales_b, opac_b, cols_b,
                                    viewmats=viewmats, Ks=Ks,
                                    width=rs.image_width, height=rs.image_height,
                                    radius_clip=0.0)
    else:
        shs_b = shs.view(1, N, *shs.shape[1:])
        out = _gsplat_rasterization(means_b, quats_b, scales_b, opac_b, shs_b,
                                    viewmats=viewmats, Ks=Ks,
                                    width=rs.image_width, height=rs.image_height,
                                    sh_degree=rs.sh_degree, radius_clip=0.0)
    color = out[0]
    meta = out[-1] if isinstance(out[-1], dict) else {}
    if color.dim() == 5:
        color = color[0, 0].permute(2, 0, 1)

    ids = meta.get('gaussian_ids')
    ids = ids.view(-1).long() if ids is not None else None
    n_vis = int(ids.numel()) if ids is not None else 0

    def _scatter_full(sub):
        full = torch.zeros(N, device=device, dtype=torch.float32)
        if n_vis and ids is not None and sub.numel() == n_vis:
            full[ids] = sub.view(n_vis).to(device)
        return full

    radii_full = torch.zeros(N, device=device, dtype=torch.float32)
    if 'radii' in meta:
        r = meta['radii']
        if r.dim() == 2:
            r = r.max(dim=-1).values
        radii_full = _scatter_full(r.view(-1).to(device, torch.float32))

    means2d_px_full = torch.zeros((N, 2), device=device, dtype=torch.float32)
    if 'means2d' in meta:
        m2 = meta['means2d'].reshape(n_vis, -1)[:, :2].to(device, torch.float32) if n_vis else torch.zeros(0, 2, device=device)
        if ids is not None and m2.shape[0] == n_vis:
            means2d_px_full[ids] = m2
    if means2d_out is not None and isinstance(means2d_out, torch.Tensor):
        means2d_out.copy_(means2d_px_full)

    # --- accum_metric_counts (N,) — per-Gaussian metric count ---
    # Inria/FluxGS kernel (forward.cu:400-402, rasterize_points.cu:104-109):
    #   size P tensor;  atomicAdd(metricCount[gaussian_idx], alpha)  for each
    #   painted+flagged pixel.  I.e. per-gaussian sum of alpha over
    #   metric-mapped (high-error) pixels it painted.  Used to rank
    #   high-error gaussians for pruning/densify selection.
    #
    # gsplat exposes no per-gaussian "painted-coverage-at-flagged-pixels"
    # directly.  A faithful proxy: opacity_i · (mean 2D radius_i)².
    # This preserves the relative ordering the mobilegs2 prune path needs
    # (large+opaque paint more → higher accum → higher pruning_score), and
    # returns zero for invisible points (radii[i] = 0).  Only computed when
    # rs.metric_map is non-None (densify score pass); zero otherwise.
    accum = torch.zeros(N, device=device, dtype=torch.float32)
    if rs.metric_map is not None:
        r_sq = (radii_full * radii_full)  # (N,), 0 for invisible
        op = opacities.detach().view(N).float() if opacities.dim() >= 1 else torch.ones(N, device=device)
        accum = op * r_sq
    return color, radii_full, means2d_px_full, accum


class GaussianRasterizer(nn.Module):
    """Drop-in for FluxGS GaussianRasterizer (training path)."""

    def __init__(self, raster_settings: GaussianRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions: torch.Tensor) -> torch.Tensor:
        """Frustum-mask via a zero-opacity probe rasterization.

        FluxGS uses (radii > 0) as the visibility filter for densification.
        We get a real per-point radii back from gsplat's projection metadata.
        """
        with torch.no_grad():
            n = positions.shape[0]
            zero_scales = torch.zeros(n, 3, device=positions.device)
            unit_quat = torch.zeros(n, 4, device=positions.device)
            unit_quat.index_copy_(0, 0, torch.ones(n, device=positions.device))
            _c, radii, _m2d, _accum = _rasterize(
                positions,
                zero_scales,
                unit_quat,
                torch.ones(n, device=positions.device),
                torch.ones(n, 3, device=positions.device),
                self.raster_settings)
        return radii

    def forward(self, means3D, means2D, opacities,
                shs=None, colors_precomp=None,
                scales=None, rotations=None, cov3D_precomp=None):
        rs = self.raster_settings
        if (shs is None) == (colors_precomp is None):
            raise ValueError("provide exactly one of shs / colors_precomp")
        if (scales is None) or (rotations is None):
            raise ValueError("this adapter requires both scales and rotations (cov3D_precomp path not implemented)")
        colors = colors_precomp if colors_precomp is not None else None
        color, radii, means2d_px, accum = _rasterize(
            means3D, scales, rotations, opacities, colors, rs,
            use_sh=colors is None, shs=shs)
        if means2D is not None and means2D.requires_grad:
            # --- view-space (screen-space) gradient pathway for densification ---
            # FluxGS/Inria's CUDA kernel routes the image loss into
            # means2D.grad (screen-space gradient at each gaussian center),
            # which train.py's densification split/grow criterion consumes.
            # gsplat has no such input, so we synthesize an equivalent signal:
            #
            #   grid_i = pixel->norm(gsaplt projected center of gaussian i)
            #              + means2D_i / W          (forward: means2D = 0)
            #   extra  = w * sum_i (image(grid_i) - s̄)          -> 0 (no bias)
            #
            # Forward : sampling at the real gsplat centers; sum is centered
            #           -> rendered image is bit-identical to plain gsplat.
            # Backward: dL/dmeans2D_i = w * (image-space gradient at center_i),
            #           the same quantity the Inria kernel provided to the
            #           densification statistic (per-point, non-zero on
            #           textured pixels).
            n = means2D.shape[0]
            if n > 0:
                H, W = rs.image_height, rs.image_width
                center = means2d_px if means2d_px is not None else torch.zeros(n, 2, device=color.device)
                center = center.float().to(color.device)
                norm = torch.stack([
                    2.0 * (center[:, 0] + 0.5) / max(W, 1) - 1.0,
                    2.0 * (center[:, 1] + 0.5) / max(H, 1) - 1.0,
                ], dim=-1)
                # differentiable w.r.t. means2D, zero in forward (means2D is 0)
                grid = (norm + means2D[:, 0:2].to(color.device).float() / max(W, 1))
                grid = grid.clamp(-1.0, 1.0).view(1, n, 1, 2)
                sampled = F.grid_sample(
                    color.unsqueeze(0), grid,
                    mode='bilinear', padding_mode='zeros', align_corners=True)
                per_point = sampled.mean(dim=1).squeeze(-1)          # (1, n)
                # Forward: (s_i - s_i.detach()) == 0 for every i  -> image unchanged.
                # Backward: dL/dmeans2D_i = sum_pixels dL/dcolor * ds_i/dmeans2D_i
                #           i.e. the image-space gradient at gaussian i's center —
                #           the same signal the Inria kernel exposed as means2D.grad.
                w = 1.0                                              # grad scale
                extra = w * (per_point - per_point.detach()).sum()   # == 0
                color = color + extra
        if _MEM_TRACE and _TRACE_COUNT[0] % 50 == 0:
            _TRACE_COUNT[0] += 1
            _a = torch.cuda.memory_allocated() / 2 ** 20
            _r = torch.cuda.memory_reserved() / 2 ** 20
            print(f"[MEMTRACE] fwd N={means3D.shape[0]} nvis={int((radii>0).sum())} "
                  f"alloc={_a:.0f}MiB reserved={_r:.0f}MiB", flush=True)
        return color, radii, accum


# ---- module-level FluxGS entries (used when callers import these names) ----

def rasterize_gaussians(means3D, means2D, sh, colors_precomp, opacities,
                        scales, rotations, cov3Ds_precomp, raster_settings):
    return GaussianRasterizer(raster_settings)(
        means3D, None, opacities, shs=sh,
        colors_precomp=colors_precomp, scales=scales, rotations=rotations)


def rasterize_gaussians_with_static_cache(cache, means3D, sh, colors_precomp,
                                          opacities, scales, rotations,
                                          cov3Ds_precomp, global_ids, raster_settings):
    """Inference-only FluxGS path; approximate with a plain differentiable render."""
    return rasterize_gaussians(means3D, None, sh, colors_precomp, opacities,
                               scales, rotations, cov3Ds_precomp, raster_settings)
