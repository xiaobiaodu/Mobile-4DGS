#!/usr/bin/env bash
# ROCm (MI300X) env setup for Mobile-4DGS (FluxGS) training — verified working
# on AMD Instinct MI300X / ROCm 7.0 (torch 2.10.0+rocm7.0 + gsplat 1.5.3 wheel).
# FWD+BWD pass, full 3DGS training loop runs end-to-end on MI300X.
set -euo pipefail

VENV=${VENV:-$HOME/gs210venv}
PY_VER=3.12

echo "[$(date)] creating venv at $VENV"
python$PY_VER -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -U pip wheel

echo "[$(date)] install torch 2.10.0+rocm7.0 (ABI matches gsplat 1.5.3 rocm wheel)"
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/rocm7.0

echo "[$(date)] install gsplat 1.5.3 rocm wheel + deps"
pip install gsplat --index-url https://pypi.amd.com --extra-index-url https://pypi.org/simple
pip install numpy jaxtyping rich ninja pillow tqdm icecream
pip install opencv-python-headless plyfile websockets

echo "[$(date)] write pure-torch shims for simple_knn and fused_ssim"
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
mkdir -p "$SITE/simple_knn" "$SITE/__pycache__"
cat > "$SITE/simple_knn/__init__.py" <<'PYEOF'
from ._C import distCUDA2
PYEOF
cat > "$SITE/simple_knn/_C.py" <<'PYEOF'
"""Pure-torch replacement for simple_knn (Inria 3DGS dep).

simple_knn._C.distCUDA2(points) -> squared nearest-neighbour distance (excluding self).
O(N^2) but chunked to cap memory.
"""
import torch


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    assert points.dim() == 2 and points.shape[1] == 3, "distCUDA2 expects (N,3)"
    x = points.float()
    n = x.shape[0]
    if n == 0:
        return torch.empty(0, device=x.device, dtype=torch.float32)
    if n < 512:
        d = torch.cdist(x, x, p=2).pow(2)
        d.fill_diagonal_(float('inf'))
        return d.min(dim=1).values
    chunk, out = 4096, torch.empty(n, device=x.device, dtype=torch.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = torch.cdist(x[s:e], x, p=2).pow(2)
        # exclude self
        rows = torch.arange(e - s, device=x.device)
        d[rows, torch.arange(s, e, device=x.device)] = float('inf')
        out[s:e] = d.min(dim=1).values
    return out
PYEOF

cat > "$SITE/fused_ssim.py" <<'PYEOF'
"""fused_ssim (torch-native) matching the FluxGS/fused-ssim submodule API.

Real API: fused_ssim(img1, img2, padding="same", train=True) -> scalar (differentiable).
We reproduce it with a windowed Gaussian SSIM (conv2d + autograd), C1=0.01^2, C2=0.03^2, 11x11 window.
"""
import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int = 11, sigma: float = 1.5, device=None, dtype=torch.float32):
    ax = torch.arange(size, device=device, dtype=dtype) - (size - 1) // 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    g = g / g.sum()
    return (g[:, None] * g[None, :]).to(dtype)


def _conv_ssim(img1, img2, padding="same"):
    C, dev = img1.shape[1], img1.device
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    k = _gaussian_kernel(11, 1.5, dev).to(img1.dtype)
    k = k.view(1, 1, 11, 11).expand(C, 1, 11, 11).contiguous()
    mu1 = F.conv2d(img1, k, padding=5, groups=C)
    mu2 = F.conv2d(img2, k, padding=5, groups=C)
    mu12 = mu1 * mu2
    s1 = F.conv2d(img1 * img1,  k, padding=5, groups=C) - mu1 * mu1
    s2 = F.conv2d(img2 * img2,  k, padding=5, groups=C) - mu2 * mu2
    s12 = F.conv2d(img1 * img2, k, padding=5, groups=C) - mu12
    ssim_map = ((2 * mu12 + C1) * (2 * s12 + C2)) / ((mu1 * mu1 + mu2 * mu2 + C1) * (s1 + s2 + C2))
    if padding == "valid":
        ssim_map = ssim_map[:, :, 5:-5, 5:-5]
    return ssim_map


def fused_ssim(img1, img2, padding="same", train=True):
    return _conv_ssim(img1, img2, padding).mean()
PYEOF

echo "[$(date)] verify install"
python - <<'PYEOF'
import torch, torch.cuda, gsplat
assert torch.cuda.is_available(), "torch.cuda not available"
print("torch:", torch.__version__, "device:", torch.cuda.get_device_name(0))
print("gsplat:", gsplat.__version__)
import simple_knn
print("simple_knn.distCUDA2:", simple_knn.distCUDA2.__module__)
import fused_ssim
print("fused_ssim:", fused_ssim.fused_ssim.__module__)
PYEOF

echo "[$(date)] done.  Activate with:  source $VENV/bin/activate"
