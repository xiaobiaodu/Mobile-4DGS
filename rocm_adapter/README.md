# Mobile-4DGS / FluxGS on AMD ROCm (MI300X)

FluxGS (Mobile-4DGS's static-3DGS base) is originally CUDA-only, via the Inria
`diff-gaussian-rasterization` + `simple_knn` + `fused_ssim` extensions. This
directory provides a **drop-in ROCm backend**: a ~250-line Python adapter that
re-implements FluxGS's `GaussianRasterizer` / `GaussianRasterizationSettings`
interface on top of **gsplat 1.5.3 (ROCm)**, plus pure-torch shims for the two
other CUDA extensions. Result: **the full FluxGS training loop runs unmodified
on AMD Instinct MI300X (ROCm 7.0)** — forward + backward + densification
(split/grow/prune) all verified working end-to-end.

## What changed (3 files, no CUDA source touched)

| File | Change |
|------|--------|
| `gaussian_renderer/__init__.py` | `from .renderer import GaussianRasterizer` → `from rocm_adapter.fluxgs_to_gsplat import ...` |
| `rocm_adapter/fluxgs_to_gsplat.py` | **New** — gsplat-backed `GaussianRasterizer` + `GaussianRasterizationSettings` |
| `rocm_adapter/setup_rocm_env.sh` | **New** — one-shot venv setup (torch 2.10.0+rocm7.0 + gsplat wheel + pure-torch `simple_knn` / `fused_ssim`) |

Everything else in `train.py`, `scene/gaussian_model.py`,
`submodules/`, etc. is **unmodified**.

## Why gsplat works as the FluxGS renderer

gsplat 1.5.3 (the `ROCm/gsplat` fork, `pypi.amd.com`) publishes ROCm wheels
with native `hip/csrc/*.hip` sources — not auto-hipified CUDA. The API
maps 1:1 onto FluxGS's Inria-convention renderer:

| FluxGS (Inria) | gsplat 1.5.3 |
|---|---|
| `means3D (N,3)` | `means (1,N,3)` |
| `quats (N,4)` (wxyz) | `quats (1,N,4)` (wxyz) — already wxyz ✓ |
| `scaling` (exp-space) | `scales = scaling` — pass through as raw linear scales |
| `colors_precomp (N,3)` | `colors (1,N,3)` |
| `opacities (N,)` | `opacities (1,N,)` |
| `viewmatrix 4x4 (w2c)` | stacked into `viewmats (1,1,4,4)` |
| `image_width/height` | `width`, `height` — no separate Ks needed |
| return `render` | `color[:,:,0:3]` |
| return `radii (N,)` | synthesized from `meta['gaussian_ids']` |
| **`means2D.grad` (densification)** | **synthesized** (see below) |

## The one hard problem: screen-space gradient (densification)

FluxGS reads `means2D.grad` — a per-gaussian **screen-space gradient** the
Inria CUDA kernel emits and routes via the (zero-in-forward) `means2D` input.
`train.py:360` reads it, `:397` builds `stats`, `:407` calls
`densify_and_prune(..., stats, ...)`, and `:421` checks
`means2D.grad != None` before deciding split vs grow. Without it, FluxGS hits
`AssertionError: rasterizer did not produce view-space gradients`.

gsplat does **not** emit this signal. The adapter synthesizes an equivalent one
with a zero-forward-bias differentiable coupling:

```python
grid   = pixel→norm(gsaplt projected center) + means2D/W    # forward: means2D==0
s_i    = image sampled at grid_i
extra  = Σ_i (s_i − s_i.detach())                            # == 0  → image unchanged
color += extra
```

* **Forward:** `s_i − s_i.detach() == 0` for every i → rendered image is bit-identical to plain gsplat.
* **Backward:** `∂L/∂means2D_i = Σ_pixels dL/dcolor · ∂s_i/∂means2D_i`, i.e. the **image-space gradient at gaussian i's projected center** — the same quantity the Inria kernel exposed to the densification statistic.

Verified: 2924/3000 rows of `means2D.grad` non-zero, `radii>0` for 2995/3000,
`means.grad` / `scaling.grad` finite, training loss decreasing 0.236→0.196 over
the first 1000 iterations on a synthetic scene at 60–110 it/s on MI300X.

## Setup (one command)

```bash
# create venv, install torch 2.10.0+rocm7.0 + gsplat 1.5.3 + pure-torch shims
bash rocm_adapter/setup_rocm_env.sh
source $HOME/gs210venv/bin/activate
```

Verified on: AMD Instinct MI300X / ROCm 7.0.51831 / Python 3.12 /
torch 2.10.0+rocm7.0 / gsplat 1.5.3+6f2f06e.

## Run (example synthetic scene)

```bash
cd mobile-4dgs
python train.py \
  --source_path /path/to/OMG4-processed-scene \
  --model_path  out/ \
  --sh_degree 0 --iterations 30000 \
  --test_iterations 1000 5000 10000 --save_iterations 30000 \
  --position_lr_init 1.6e-4 --position_lr_final 1.6e-5 \
  --convert_SHs_python --resolution -1
```

## What is *not* covered yet (out of scope of this backend)

* **4D dynamics** (velocity field, PDE / time advection) — gsplat 1.5.3 has no
  temporal primitives; FluxGS's dynamic-4D path is unchanged and still needs
  its own ROCm work.
* **Antialiasing** (FluxGS `--antialiasing` flag) — not currently routed.
* **MV / multi-view** renderer variant.
* **`--ip`/`--port`** websockets training mode.

The static-3DGS path (the one FluxGS uses by default) is fully working.

## Files

```
rocm_adapter/
├── fluxgs_to_gsplat.py   # gsplat backend (GaussianRasterizer + Settings drop-in)
├── setup_rocm_env.sh     # one-shot venv + wheel + pure-torch shim install
├── README.md             # this file
└── IMPLEMENTATION.md     # full implementation record (design, contract gaps, verification)
```
