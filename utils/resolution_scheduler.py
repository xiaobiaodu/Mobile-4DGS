"""DashGaussian-style frequency-guided resolution scheduling.

Training starts on a low-resolution proxy of the training views and the render
resolution is raised back to the full camera resolution along a schedule that is
derived from the scene itself (DashGaussian, arXiv:2503.18402, sec. 4.2 and
supplementary eq. (6), (7)).

The scene statistic is the DFT magnitude energy of the training views

    X(F) = (1/N) * sum_n sum_{i,j} ||F^n(i,j)||_2

with ``F_r = DFT(I_r)`` for the image downsampled by ``r``.  The schedules uses

    s_{r_i} = S * ln(X(F_{r_i}) / X(F_{r_m})) / ln(X(F) / X(F_{r_m}))

where ``r_m`` is the largest downsampling factor whose energy dropped to
``X(F)/a`` (``a`` is ``--multiscale_max_scale``, ``S`` the iteration that
reaches full resolution), and the resolution between two consecutive switches is
interpolated with the inverse square of the two factors.  That continuous factor
is discretized to an integer by :meth:`ResolutionSchedule.scale_for`.

Deviation from the paper worth knowing about: the sums are taken on an
orthonormal DFT (``norm="ortho"``).  With the plain (backward) transform the DC
bin of a box-downsampled image decays as ``1/r^2`` and dominates the sum, which
makes ``X(F_r) ~ X(F)/r^2`` for any scene and collapses ``a`` into a constant
``sqrt(a)`` downsampling factor.  Under the orthonormal transform the retained
band scales as ``1/r``, so ``a = 4`` really does start at ``1/4`` resolution,
``a = 1`` still disables the schedule, and the resulting factor stays
scene-adaptive: scenes whose energy is already concentrated in low frequencies
tolerate a stronger downsample.

This module contains no densification logic; it only produces the render
resolution used by the training step.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["ResolutionSchedule"]


def _to_gray(image: torch.Tensor) -> torch.Tensor:
    """Collapse an RGB(A) image in [0, 1] to a single float channel."""
    channels = image[:3] if image.shape[0] >= 3 else image
    return channels.float().mean(dim=0, keepdim=False)


def _resize(gray: torch.Tensor, size) -> torch.Tensor:
    if tuple(gray.shape[-2:]) == (int(size[0]), int(size[1])):
        return gray
    return F.interpolate(
        gray[None, None],
        size=(int(size[0]), int(size[1])),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0, 0]


def _box_downsample(gray: torch.Tensor, factor: int) -> torch.Tensor:
    """Area-average a single channel by an integer factor (box filter)."""
    height, width = gray.shape[-2:]
    target = (max(1, height // factor), max(1, width // factor))
    if target == (height, width):
        return gray
    return F.interpolate(gray[None, None], size=target, mode="area")[0, 0]


def _select_view_indices(cameras, num_views: int, seed: int) -> List[int]:
    """Deterministically pick a small, time-stratified subset of training views.

    Dynamic datasets expose hundreds of camera-frame samples, so the subset is
    drawn one per temporal chunk when timestamps are available.  Static
    datasets fall back to a uniform random sample.  A local RandomState is used
    so the global RNG stream of the training run is left untouched.
    """
    count = len(cameras)
    budget = max(1, min(int(num_views), count))
    if budget >= count:
        return list(range(count))

    rng = np.random.RandomState(int(seed) % (2**31 - 1))
    times = np.asarray(
        [
            np.nan if getattr(camera, "time", None) is None else float(camera.time)
            for camera in cameras
        ],
        dtype=np.float64,
    )
    if np.all(np.isfinite(times)) and float(times.max()) > float(times.min()):
        chunk = count / float(budget)
        order = np.argsort(times, kind="stable")
        selected = set()
        for index in range(budget):
            start = int(index * chunk)
            span = max(1, int(round(chunk)))
            offset = int(rng.randint(0, span))
            selected.add(int(order[min(count - 1, start + offset)]))
        return sorted(selected)

    return sorted(int(i) for i in rng.choice(count, budget, replace=False))


def _base_resolution(cameras, indices: Sequence[int]):
    """Median loaded resolution of the sampled views.

    The per-view loaded resolution depends on ``--resolution`` (and on the
    per-image rescaling that happens when ``--resolution -1``), so the energy
    curve is always measured on one common grid.
    """
    heights = np.asarray([int(cameras[i].image_height) for i in indices])
    widths = np.asarray([int(cameras[i].image_width) for i in indices])
    return int(np.median(heights)), int(np.median(widths))


def _measure_energy_curve(cameras, indices, base_hw, max_scale: int) -> Optional[List[float]]:
    """Mean DFT magnitude energy of the sampled views at each integer factor.

    Index ``r`` of the returned list holds ``X(F_r)``; index 0 mirrors index 1.
    The curve is forced to be non-increasing so the inversion below is stable.
    """
    energies = [0.0] * (max_scale + 1)
    used = 0
    for index in indices:
        try:
            image = cameras[index].original_image
        except Exception as error:  # pragma: no cover - dataset dependent
            print(f"[multi-scale] skipping view {index} while measuring energy: {error}")
            continue
        if image is None:
            continue
        gray = _to_gray(image)
        # The spectrum is only ever reduced to a scalar, so running it on the
        # GPU (when available) keeps the one-off measurement cheap.
        if torch.cuda.is_available() and not gray.is_cuda:
            gray = gray.to("cuda", non_blocking=True)
        gray = _resize(gray, base_hw)
        for factor in range(1, max_scale + 1):
            sample = gray if factor == 1 else _box_downsample(gray, factor)
            spectrum = torch.fft.fft2(sample, norm="ortho")
            energies[factor] += float(spectrum.abs().sum())
        used += 1
        del image, gray
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if used == 0:
        return None

    for factor in range(1, max_scale + 1):
        energies[factor] /= float(used)
    energies[0] = energies[1]
    for factor in range(2, max_scale + 1):
        energies[factor] = min(energies[factor], energies[factor - 1])
    return energies


def _invert_energy(energies: Sequence[float], target: float, max_scale: int) -> float:
    """Continuous downsampling factor whose energy equals ``target``.

    ``energies`` is non-increasing on ``1..max_scale``.  The bracketing samples
    are interpolated in log-log space so a power-law decay (the usual case) is
    inverted exactly.
    """
    if target >= energies[1]:
        return 1.0
    if target <= energies[max_scale]:
        return float(max_scale)
    for factor in range(1, max_scale):
        high, low = energies[factor], energies[factor + 1]
        if high >= target > low:
            if high <= low:
                return float(factor)
            weight = math.log(high / target) / math.log(high / low)
            return float(factor) * ((factor + 1) / factor) ** weight
    return float(max_scale)


class ResolutionSchedule:
    """Iteration -> integer downsampling factor lookup built once per run."""

    def __init__(
        self,
        factors: Sequence[float],
        switch_iters: Sequence[int],
        horizon: int,
        base_resolution,
        energies: Sequence[float],
        max_scale: int,
        num_views: int,
        max_scale_request: float,
    ):
        if len(factors) != len(switch_iters) or len(factors) < 2:
            raise ValueError("resolution schedule needs at least two levels")
        self.factors = [float(f) for f in factors]
        self.switch_iters = [int(s) for s in switch_iters]
        self.horizon = int(horizon)
        self.base_resolution = (int(base_resolution[0]), int(base_resolution[1]))
        self.energies = [float(e) for e in energies]
        self.max_scale = int(max_scale)
        self.num_views = int(num_views)
        self.max_scale_request = float(max_scale_request)
        self.full_resolution_iter = self._find_full_resolution_iter()

    # -- construction ----------------------------------------------------
    @classmethod
    def build(
        cls,
        cameras,
        horizon: int,
        max_scale: float = 4.0,
        num_levels: int = 5,
        num_views: int = 16,
        search_max_scale: int = 8,
        seed: int = 0,
    ) -> Optional["ResolutionSchedule"]:
        """Derive the schedule from the training views.

        Returns ``None`` when no resolution change is warranted (``a <= 1``, no
        usable views, or a schedule that would never leave full resolution);
        callers must treat that as "multi-scale disabled".
        """
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError(f"resolution schedule horizon must be positive, got {horizon}")
        max_scale = float(max_scale)
        if max_scale <= 1.0:
            return None
        if len(cameras) == 0:
            return None

        search_max = max(2, int(search_max_scale))
        indices = _select_view_indices(cameras, num_views, seed)
        base_hw = _base_resolution(cameras, indices)
        energies = _measure_energy_curve(cameras, indices, base_hw, search_max)
        if energies is None:
            return None

        full_energy = energies[1]
        if not math.isfinite(full_energy) or full_energy <= 0.0:
            return None
        target = full_energy / max_scale
        highest_factor = _invert_energy(energies, target, search_max)
        if highest_factor <= 1.0 + 1e-6:
            return None

        levels = max(2, int(num_levels))
        # Energy levels are sampled uniformly between "a times cheaper" and the
        # full-resolution energy; the paper's a defines both ends.
        energies_at_levels = [
            target + (full_energy - target) * (index / float(levels - 1))
            for index in range(levels)
        ]
        factors = [_invert_energy(energies, energy, search_max) for energy in energies_at_levels]
        # Guard the monotonicity of the schedule: resolution only ever rises.
        for index in range(1, levels):
            factors[index] = min(factors[index], factors[index - 1])
        factors[-1] = 1.0

        # Eq. (7): the share of the horizon spent at each level is the log
        # energy ratio between that level and the lowest one.
        denominator = math.log(full_energy / target)
        switch_iters = [
            int(round(horizon * math.log(energy / target) / denominator))
            for energy in energies_at_levels
        ]
        # The first and last switches are pinned so the schedule spans exactly
        # [0, horizon] and never regresses on rounding.
        switch_iters[0] = 0
        switch_iters[-1] = horizon
        for index in range(1, levels):
            switch_iters[index] = max(switch_iters[index], switch_iters[index - 1])

        schedule = cls(
            factors=factors,
            switch_iters=switch_iters,
            horizon=horizon,
            base_resolution=base_hw,
            energies=energies,
            max_scale=search_max,
            num_views=len(indices),
            max_scale_request=max_scale,
        )
        if schedule.scale_for(0) <= 1:
            # The energy curve dropped too quickly for any integer factor to be
            # worth it; report "disabled" instead of pretending to schedule.
            return None
        return schedule

    # -- query -----------------------------------------------------------
    def factor_at(self, iteration: int) -> float:
        """Continuous downsampling factor (>= 1) for this iteration."""
        factors = self.factors
        if iteration <= self.switch_iters[0]:
            return factors[0]
        if iteration >= self.horizon:
            return 1.0
        for index in range(len(factors) - 1):
            lower, upper = self.switch_iters[index], self.switch_iters[index + 1]
            if iteration < upper:
                span = upper - lower
                if span <= 0:
                    return min(factors[index], factors[index + 1])
                weight = (iteration - lower) / float(span)
                # Interpolate the inverse square of the two neighbouring
                # factors and convert back; this is the paper's interpolation.
                inverse_square = (
                    (1.0 - weight) / (factors[index] ** 2)
                    + weight / (factors[index + 1] ** 2)
                )
                return 1.0 / math.sqrt(max(inverse_square, 1e-12))
        return 1.0

    def scale_for(self, iteration: int) -> int:
        """Integer downsampling factor (>= 1) used by the training step.

        The paper floors the interpolated factor.  Rounding is used instead:
        flooring collapses the whole schedule back to full resolution as soon
        as the measured ``r_m`` sits close to 2, which is the common case for
        textured scenes, and would silently turn the schedule into a no-op.
        """
        return max(1, int(self.factor_at(iteration) + 0.5))

    def _find_full_resolution_iter(self) -> int:
        for iteration in range(0, self.horizon + 1):
            if self.scale_for(iteration) <= 1:
                return iteration
        return self.horizon

    # -- reporting -------------------------------------------------------
    def summary(self) -> str:
        effective = [self.scale_for(iteration) for iteration in self.switch_iters]
        base = f"{self.base_resolution[0]}x{self.base_resolution[1]}"
        lines = [
            (
                "Multi-scale training enabled (frequency-guided resolution schedule): "
                f"a={self.max_scale_request:g}, base={base}, "
                f"views={self.num_views}, horizon={self.horizon}"
            ),
            (
                "  energy curve X(F_r): "
                + ", ".join(f"r={r}:{self.energies[r]:.3g}" for r in range(1, self.max_scale + 1))
            ),
            (
                "  levels (iteration: raw r -> downsample factor): "
                + ", ".join(
                    f"{self.switch_iters[i]}: {self.factors[i]:.2f} -> {effective[i]}x"
                    for i in range(len(self.factors))
                )
            ),
            (
                f"  full resolution from iteration {self.full_resolution_iter} "
                f"of the requested {self.horizon}"
            ),
            (
                "  note: densification thresholds and scoring are unchanged, but the "
                "screen-space statistics collected before full resolution are smaller "
                "because the training render is downsampled."
            ),
        ]
        return "\n".join(lines)
