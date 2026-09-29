import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.general_utils import safe_state
import torchvision
import time
import json
import math
import statistics
from gaussian_renderer import (
    ExactStaticCacheRenderer,
    fixed_camera_cache_signature,
    render_fluxgs,
)


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _validate_fixed_camera_sequence(views, pipeline, mult):
    if not views:
        raise ValueError("Cannot build a static cache for an empty view sequence")
    expected = fixed_camera_cache_signature(views[0], pipeline, mult)
    for index, view in enumerate(views[1:], start=1):
        signature = fixed_camera_cache_signature(view, pipeline, mult)
        if signature != expected:
            raise RuntimeError(
                "Exact static caching requires a fixed camera, but view "
                f"{index} has different extrinsics/intrinsics or resolution"
            )
    return expected


def _time_renderer(views, render_fn, repeats):
    repeat_wall_ms = []
    frame_cuda_ms = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        starts = []
        ends = []
        wall_start = time.perf_counter()
        for view in views:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            render_fn(view)
            end.record()
            starts.append(start)
            ends.append(end)
        torch.cuda.synchronize()
        repeat_wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        frame_cuda_ms.extend(
            float(start.elapsed_time(end)) for start, end in zip(starts, ends)
        )
    wall_ms = statistics.median(repeat_wall_ms)
    return {
        "wall_total_ms": wall_ms,
        "fps": len(views) * 1000.0 / wall_ms if wall_ms > 0 else float("inf"),
        "cuda_mean_ms": statistics.fmean(frame_cuda_ms),
        "cuda_p50_ms": _percentile(frame_cuda_ms, 50),
        "cuda_p95_ms": _percentile(frame_cuda_ms, 95),
        "repeat_wall_ms": repeat_wall_ms,
    }


def benchmark_static_cache(
    views,
    gaussians,
    pipeline,
    background,
    mult,
    warmup=10,
    repeats=3,
    max_frames=0,
    json_path="",
):
    """Benchmark full sorting against exact fixed-camera static caching."""
    if not gaussians.dynamic_enabled:
        raise ValueError("--benchmark_static_cache requires a dynamic model")
    if max_frames > 0:
        views = list(views[:max_frames])
    else:
        views = list(views)
    if not views:
        raise ValueError("No test views are available for the cache benchmark")
    _validate_fixed_camera_sequence(views, pipeline, mult)
    for view in views:
        view.load_cam_parm_to_device(torch.device("cuda"))

    cache_renderer = ExactStaticCacheRenderer(
        gaussians, pipeline, background, mult
    )
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    torch.cuda.synchronize()
    build_wall_start = time.perf_counter()
    cache_renderer.build(views[0])
    torch.cuda.synchronize()
    cache_build_wall_ms = (time.perf_counter() - build_wall_start) * 1000.0
    cache_allocated_bytes = max(
        0, torch.cuda.memory_allocated() - allocated_before
    )
    cache_peak_bytes = torch.cuda.max_memory_allocated()

    def baseline(view):
        return render_fluxgs(
            view,
            gaussians,
            pipeline,
            background,
            mult,
            gaussian_subset="all",
            dynamic_gate_stochastic=False,
        )["render"]

    def cached(view):
        return cache_renderer.render(view)["render"]

    warmup_count = min(max(int(warmup), 0), len(views))
    for view in views[:warmup_count]:
        baseline(view)
        cached(view)
    torch.cuda.synchronize()

    baseline_timing = _time_renderer(views, baseline, max(1, int(repeats)))
    cached_timing = _time_renderer(views, cached, max(1, int(repeats)))

    max_abs = 0.0
    absolute_sum = 0.0
    squared_sum = 0.0
    element_count = 0
    identical_frames = 0
    for view in tqdm(views, desc="Validating cache", leave=False):
        reference = baseline(view)
        candidate = cached(view)
        difference = (reference - candidate).float()
        frame_max = float(difference.abs().max().item())
        max_abs = max(max_abs, frame_max)
        if frame_max == 0.0:
            identical_frames += 1
        absolute_sum += float(difference.abs().sum().item())
        squared_sum += float(difference.square().sum().item())
        element_count += difference.numel()
    mean_abs = absolute_sum / max(element_count, 1)
    rmse = math.sqrt(squared_sum / max(element_count, 1))
    equivalence_psnr = (
        float("inf") if rmse == 0.0 else -20.0 * math.log10(rmse)
    )

    steady_speedup = baseline_timing["wall_total_ms"] / cached_timing["wall_total_ms"]
    build_ms = float(cache_renderer.build_ms)
    amortized_cached_ms = cached_timing["wall_total_ms"] + cache_build_wall_ms
    amortized_speedup = baseline_timing["wall_total_ms"] / amortized_cached_ms
    saved_per_frame_ms = (
        baseline_timing["wall_total_ms"] - cached_timing["wall_total_ms"]
    ) / len(views)
    break_even_frames = (
        math.ceil(cache_build_wall_ms / saved_per_frame_ms)
        if saved_per_frame_ms > 0 else None
    )
    dynamic_count = int(cache_renderer.dynamic_indices.numel())
    report = {
        "frames": len(views),
        "warmup_frames": warmup_count,
        "repeats": max(1, int(repeats)),
        "point_count": cache_renderer.point_count,
        "static_count": int(cache_renderer.static_indices.numel()),
        "dynamic_count": dynamic_count,
        "dynamic_fraction": dynamic_count / max(cache_renderer.point_count, 1),
        "cache_build_ms": build_ms,
        "cache_build_wall_ms": cache_build_wall_ms,
        "cache_resident_mib": cache_renderer.cache.resident_bytes / (1024.0 ** 2),
        "cache_allocator_delta_mib": cache_allocated_bytes / (1024.0 ** 2),
        "cache_peak_allocated_mib": cache_peak_bytes / (1024.0 ** 2),
        "baseline": baseline_timing,
        "cached": cached_timing,
        "steady_speedup": steady_speedup,
        "amortized_speedup": amortized_speedup,
        "break_even_frames": break_even_frames,
        "correctness": {
            "identical_frames": identical_frames,
            "bitwise_identical": identical_frames == len(views),
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "rmse": rmse,
            "cache_vs_baseline_psnr": equivalence_psnr,
            "gt_psnr_ssim_lpips_unchanged": identical_frames == len(views),
        },
    }

    print(
        "Exact static cache benchmark: "
        f"{report['static_count']:,} static / {dynamic_count:,} dynamic, "
        f"build {cache_build_wall_ms:.2f} ms wall ({build_ms:.2f} ms CUDA)"
    )
    print(
        f"  baseline {baseline_timing['fps']:.2f} FPS | "
        f"cached {cached_timing['fps']:.2f} FPS | "
        f"steady {steady_speedup:.3f}x | amortized {amortized_speedup:.3f}x"
    )
    print(
        f"  correctness max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}, "
        f"cache-vs-baseline PSNR={equivalence_psnr:.3f} dB"
    )
    if identical_frames == len(views):
        print("  bitwise identical: GT PSNR / SSIM / LPIPS are unchanged")
    elif max_abs > 1e-6:
        print("  WARNING: cached output differs materially from the full renderer")
    if json_path:
        output_dir = os.path.dirname(os.path.abspath(json_path))
        makedirs(output_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, allow_nan=True)
        print(f"  report: {json_path}")
    return report


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mult, render_split=False, exact_static_cache=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    static_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_static")
    dynamic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_dynamic")


    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if render_split:
        makedirs(static_path, exist_ok=True)
        makedirs(dynamic_path, exist_ok=True)


    cache_renderer = None
    render_events = []
    if exact_static_cache:
        if render_split:
            raise ValueError("Exact cached rendering cannot emit diagnostic split passes")
        _validate_fixed_camera_sequence(views, pipeline, mult)
        cache_renderer = ExactStaticCacheRenderer(
            gaussians, pipeline, background, mult
        )
        views[0].load_cam_parm_to_device(torch.device("cuda"))
        cache_renderer.build(views[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        view.load_cam_parm_to_device(torch.device("cuda"))

        if cache_renderer is None:
            rendering = render_fluxgs(
                view, gaussians, pipeline, background, mult, gaussian_subset="all"
            )["render"]
        else:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            rendering = cache_renderer.render(view)["render"]
            end.record()
            render_events.append((start, end))
 
        
        gt = view.original_image[0:3, :, :]
        filename = '{0:05d}.png'.format(idx)
        torchvision.utils.save_image(rendering, os.path.join(render_path, filename))
        torchvision.utils.save_image(gt, os.path.join(gts_path, filename))

        if render_split:
            static_rendering = render_fluxgs(
                view,
                gaussians,
                pipeline,
                background,
                mult,
                gaussian_subset="static",
            )["render"]
            dynamic_rendering = render_fluxgs(
                view,
                gaussians,
                pipeline,
                background,
                mult,
                gaussian_subset="dynamic",
            )["render"]
            torchvision.utils.save_image(
                static_rendering, os.path.join(static_path, filename)
            )
            torchvision.utils.save_image(
                dynamic_rendering, os.path.join(dynamic_path, filename)
            )

    if cache_renderer is not None and render_events:
        torch.cuda.synchronize()
        frame_ms = [
            float(start.elapsed_time(end)) for start, end in render_events
        ]
        mean_ms = statistics.fmean(frame_ms)
        print(
            f"Exact static cache: build {cache_renderer.build_ms:.2f} ms, "
            f"steady {1000.0 / mean_ms:.2f} FPS, "
            f"resident {cache_renderer.cache.resident_bytes / (1024.0 ** 2):.2f} MiB"
        )
    

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    mult=0.5,
    split_dynamic=True,
    decode=False,
    exact_static_cache=False,
    benchmark_cache=False,
    benchmark_warmup=10,
    benchmark_repeats=3,
    benchmark_frames=0,
    benchmark_json="",
):

    with torch.no_grad():
        dataset.data_device = "cpu"

        # Measure loading memory
        torch.cuda.reset_peak_memory_stats()
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset,
            gaussians,
            load_iteration=iteration,
            shuffle=False,
            decode=decode,
        )
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_split = (
            split_dynamic
            and gaussians.dynamic_enabled
            and not exact_static_cache
            and not benchmark_cache
        )
        if render_split:
            dynamic_count = int(gaussians.get_dynamic_mask().sum().item())
            point_count = gaussians.get_xyz.shape[0]
            print(
                f"Rendering strict split: {point_count - dynamic_count:,} static / "
                f"{dynamic_count:,} dynamic Gaussians"
            )

        test_views = scene.getTestCameras()
        if benchmark_cache:
            benchmark_static_cache(
                test_views,
                gaussians,
                pipeline,
                background,
                mult,
                warmup=benchmark_warmup,
                repeats=benchmark_repeats,
                max_frames=benchmark_frames,
                json_path=benchmark_json,
            )
            return

        render_set(
            dataset.model_path,
            "test",
            scene.loaded_iter,
            test_views,
            gaussians,
            pipeline,
            background,
            mult,
            render_split=render_split,
            exact_static_cache=exact_static_cache,
        )


        

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mult", type=float, default=0.5)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument(
        "--skip_split_render",
        action="store_true",
        help="Do not emit renders_static/renders_dynamic for dynamic models",
    )
    parser.add_argument(
        "--exact_static_cache",
        action="store_true",
        help=(
            "Use inference-only exact static projection/sort caching for a "
            "fixed-camera dynamic sequence"
        ),
    )
    parser.add_argument(
        "--benchmark_static_cache",
        action="store_true",
        help=(
            "Benchmark full sorting against exact fixed-camera caching and "
            "exit without writing PNGs"
        ),
    )
    parser.add_argument("--benchmark_warmup", type=int, default=10)
    parser.add_argument("--benchmark_repeats", type=int, default=3)
    parser.add_argument(
        "--benchmark_frames",
        type=int,
        default=0,
        help="Limit benchmark frames; 0 uses the complete test sequence",
    )
    parser.add_argument(
        "--benchmark_json",
        type=str,
        default="",
        help="Optional path for the machine-readable benchmark report",
    )

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        mult=args.mult,
        split_dynamic=not args.skip_split_render,
        decode=args.decode,
        exact_static_cache=args.exact_static_cache,
        benchmark_cache=args.benchmark_static_cache,
        benchmark_warmup=args.benchmark_warmup,
        benchmark_repeats=args.benchmark_repeats,
        benchmark_frames=args.benchmark_frames,
        benchmark_json=args.benchmark_json,
    )
