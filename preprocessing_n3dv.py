#!/usr/bin/env python3
"""Generate FreeTimeGS initialization for every N3DV scene.

The official FreeTimeGS initialization first obtains dense multi-view matches
with RoMa, triangulates a point cloud at each selected time, and initializes
velocity by nearest-neighbor matching against the following frame.

Camera input may be either the original ``camXX.mp4`` videos or extracted
``camXX/images`` directories. By default, video input is sequentially extracted
to clip-local, lossless PNGs before reconstruction so both preprocessing and
training can use the same image layout as the other N3DV scenes.

Outputs for each scene:
  camXX/images/XXXX.png (for video input, unless --no-extract_images)
  freetime_preprocess/points3d_frameXXXXXX.npy
  freetime_preprocess/colors_frameXXXXXX.npy
  freetime_preprocess/point_clouds/frameXXXXXX.ply
  freetime_velocity_init.npz
  freetime_velocity_init.ply

With --blender_output, the same run also writes the legacy Blender/OMG4 files:
  images/camXX_XXXX.png
  transforms_train.json / transforms_test.json
  points3d.ply
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from plyfile import PlyElement, PlyData
from scipy.spatial import cKDTree
from tqdm import tqdm

from utils.n3dv_utils import load_n3dv_poses


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PREPROCESS_SCHEMA_VERSION = 2
CACHE_METADATA_SUFFIX = ".cache.json"
EXTRACTED_IMAGES_SCHEMA_VERSION = 2
BLENDER_OUTPUT_MANIFEST_NAME = "blender_output_manifest.json"
EXTRACTED_IMAGES_MANIFEST_NAME = "extracted_images_manifest.json"
EXTRACTED_IMAGES_OUTPUT_PATTERN = "camXX/images/{local_frame:04d}.png"


@dataclass
class N3DVRig:
    scene_path: Path
    camera_names: list
    frame_paths: list | None
    video_paths: list | None
    frame_count_value: int
    projections: np.ndarray
    world_to_cameras: np.ndarray
    camera_centers: np.ndarray
    width: int
    height: int
    focal: float
    fps: float | None = None
    extracted_frame_paths: list | None = None
    extracted_frame_start: int | None = None

    @property
    def frame_count(self):
        return self.frame_count_value

    @property
    def source_kind(self):
        return "videos" if self.video_paths is not None else "images"


def _json_fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_signature(path, hash_contents=False):
    path = Path(path)
    signature = {"path": str(path.resolve())}
    if not path.is_file():
        signature["missing"] = True
        return signature
    stat = path.stat()
    signature.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if hash_contents:
        digest = hashlib.sha256()
        with open(path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        signature["sha256"] = digest.hexdigest()
    return signature


def centered_camera_rig_radius(camera_centers):
    camera_centers = np.asarray(camera_centers, dtype=np.float64)
    if len(camera_centers) == 0:
        raise ValueError("Cannot derive a scene scale without camera centers")
    center = np.median(camera_centers, axis=0)
    radii = np.linalg.norm(camera_centers - center, axis=1)
    positive = radii[np.isfinite(radii) & (radii > 0)]
    if len(positive) == 0:
        raise ValueError("Camera centers do not span a non-zero scene scale")
    return float(np.median(positive))


def build_scene_identity(rig):
    """Return a stable, scene-specific identity used by caches and NPZ metadata."""
    if rig.video_paths is not None:
        source_signatures = [_file_signature(path) for path in rig.video_paths]
    else:
        source_signatures = []
        for paths in rig.frame_paths:
            # Exact per-frame signatures are added to each cache entry. The
            # endpoints here cheaply detect a replaced or re-extracted source.
            source_signatures.append({
                "first": _file_signature(paths[0]),
                "last": _file_signature(paths[-1]),
                "frame_count": len(paths),
            })
    identity = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "scene_name": rig.scene_path.name,
        "scene_path": str(rig.scene_path.resolve()),
        "poses_bounds": _file_signature(rig.scene_path / "poses_bounds.npy", hash_contents=True),
        "source_kind": rig.source_kind,
        "source_frame_count": int(rig.frame_count),
        "camera_names": list(rig.camera_names),
        "width": int(rig.width),
        "height": int(rig.height),
        "focal": float(rig.focal),
        "fps": float(rig.fps) if rig.fps is not None else None,
        "source_signatures": source_signatures,
    }
    identity["fingerprint"] = _json_fingerprint(identity)
    return identity


def build_reconstruction_context(
    rig,
    camera_pairs,
    included_camera_indices,
    excluded_camera_names,
    args,
):
    scene_identity = build_scene_identity(rig)
    try:
        romatch_version = importlib_metadata.version("romatch")
    except importlib_metadata.PackageNotFoundError:
        romatch_version = "unknown"
    settings = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "roma_model": args.roma_model,
        "romatch_version": romatch_version,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "device": str(args.device),
        "matches_per_pair": int(args.matches_per_pair),
        "min_certainty": float(args.min_certainty),
        "ransac_threshold": float(args.ransac_threshold),
        "max_reprojection_error": float(args.max_reprojection_error),
        "scene_radius_factor": float(args.scene_radius_factor),
        "voxel_size": float(args.voxel_size),
        "max_points_per_frame": int(args.max_points_per_frame),
        "seed": int(args.seed),
        "deterministic": bool(getattr(args, "deterministic", True)),
        "included_cameras": [rig.camera_names[index] for index in included_camera_indices],
        "excluded_cameras": list(excluded_camera_names),
        "camera_pairs": [
            [rig.camera_names[camera_a], rig.camera_names[camera_b]]
            for camera_a, camera_b in camera_pairs
        ],
    }
    context = {"scene": scene_identity, "reconstruction_settings": settings}
    context["fingerprint"] = _json_fingerprint(context)
    return context


def _frame_source_signatures(rig, frame_index, camera_indices):
    if rig.video_paths is not None:
        return [_file_signature(rig.video_paths[index]) for index in camera_indices]
    return [_file_signature(rig.frame_paths[index][frame_index]) for index in camera_indices]


def frame_cache_provenance(rig, frame_index, camera_indices, reconstruction_context):
    provenance = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "scene_name": rig.scene_path.name,
        "scene_path": str(rig.scene_path.resolve()),
        "source_kind": rig.source_kind,
        "scene_fingerprint": reconstruction_context["scene"]["fingerprint"],
        "reconstruction_fingerprint": reconstruction_context["fingerprint"],
        "reconstruction_settings": reconstruction_context["reconstruction_settings"],
        "frame_index": int(frame_index),
        "source_frames": _frame_source_signatures(rig, frame_index, camera_indices),
    }
    provenance["fingerprint"] = _json_fingerprint(provenance)
    return provenance


def derived_seed(base_seed, *identity_parts):
    payload = {"base_seed": int(base_seed), "identity": [str(part) for part in identity_parts]}
    # OpenCV requires a signed 32-bit RNG seed.
    return int(_json_fingerprint(payload)[:8], 16) % (2**31 - 1)


def seed_random_generators(seed, deterministic=False):
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Triangulate RoMa point clouds and build FreeTimeGS initialization for N3DV."
    )
    parser.add_argument("--dataset_root", type=Path, default=Path("../datasets/N3DV"))
    parser.add_argument(
        "--input_source",
        choices=["auto", "videos", "images"],
        default="auto",
        help=(
            "Choose original camXX.mp4 videos or extracted camXX/images. Auto keeps "
            "complete videos as origin provenance when available, then falls back "
            "to complete image folders."
        ),
    )
    parser.add_argument(
        "--extract_images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For video input, sequentially extract the selected clip to lossless "
            "camXX/images/0000.png... files for preprocessing and training. This "
            "also runs with --build_init_only; pass --no-extract_images to leave "
            "the image layout untouched."
        ),
    )
    parser.add_argument(
        "--blender_output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also create the legacy Blender/OMG4 training layout (images/, "
            "transforms_train.json, transforms_test.json, and points3d.ply) from "
            "the same extracted frames and RoMa reconstruction. This replaces the "
            "need to run n3v2blender.py separately."
        ),
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Scene directory names. Omit to process every valid N3DV scene.",
    )
    parser.add_argument("--frame_start", type=int, default=0)
    parser.add_argument("--frame_end", type=int, default=299)
    parser.add_argument(
        "--keyframe_step",
        type=int,
        default=1,
        help=(
            "Stride between velocity source frames. The default uses every "
            "consecutive transition in the selected clip."
        ),
    )
    parser.add_argument("--pairs_per_camera", type=int, default=2)
    parser.add_argument("--matches_per_pair", type=int, default=4096)
    parser.add_argument("--min_certainty", type=float, default=0.2)
    parser.add_argument("--ransac_threshold", type=float, default=1.0)
    parser.add_argument("--max_reprojection_error", type=float, default=3.0)
    parser.add_argument(
        "--max_velocity_distance",
        type=float,
        default=None,
        help=(
            "Maximum one-frame correspondence distance in scene units. By default, "
            "derive a separate threshold for each scene from its centered camera-rig radius."
        ),
    )
    parser.add_argument(
        "--velocity_distance_scene_fraction",
        type=float,
        default=0.05,
        help="Automatic velocity threshold as a fraction of the scene camera-rig radius.",
    )
    parser.add_argument(
        "--velocity_match_mode",
        choices=["mutual", "forward"],
        default="mutual",
        help=(
            "Use reciprocal nearest neighbors by default. 'forward' reproduces the "
            "one-way FreeTimeGsVanilla correspondence rule."
        ),
    )
    parser.add_argument(
        "--normalized_velocity_cap",
        type=float,
        default=0.0,
        help=(
            "Optional normalized-time velocity cap used by training. Zero (default) "
            "disables cap alignment; a positive value limits the automatic matching "
            "distance and reports clipping diagnostics."
        ),
    )
    parser.add_argument("--scene_radius_factor", type=float, default=2.0)
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--max_points_per_frame", type=int, default=100_000)
    parser.add_argument("--max_init_points", type=int, default=2_000_000)
    parser.add_argument(
        "--roma_model",
        choices=["outdoor", "indoor", "tiny"],
        default="outdoor",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--held_out_camera",
        default="cam00",
        help=(
            "N3DV evaluation camera excluded from reconstruction unless explicitly "
            "included. This must match the first camera in poses_bounds.npy (cam00 "
            "for the standard dataset)."
        ),
    )
    parser.add_argument(
        "--include_eval_camera",
        action="store_true",
        help="Include --held_out_camera in reconstruction (not evaluation-safe).",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Seed Python, NumPy, OpenCV, and Torch/RoMa and request deterministic Torch kernels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--build_init_only",
        action="store_true",
        help=(
            "Skip RoMa and rebuild only freetime_velocity_init.npz/PLY from validated "
            "per-frame reconstruction caches. For video input, the selected clip is "
            "still extracted by default; use --no-extract_images to skip extraction."
        ),
    )
    parser.add_argument(
        "--trust_legacy_cache",
        action="store_true",
        help=(
            "Allow --build_init_only to consume old cache arrays that have no provenance "
            "sidecars. The resulting NPZ is marked unverified and not evaluation-safe."
        ),
    )
    parser.add_argument("--no_ply", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def discover_scenes(dataset_root, requested_scenes=None):
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"N3DV dataset root does not exist: {dataset_root}")
    if requested_scenes:
        candidates = [dataset_root / name for name in requested_scenes]
    else:
        candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir())

    scenes = []
    for path in candidates:
        if not path.is_dir():
            raise FileNotFoundError(f"Requested N3DV scene does not exist: {path}")
        has_images = any(candidate.is_dir() for candidate in path.glob("cam*/images"))
        has_videos = any(candidate.is_file() for candidate in path.glob("cam*.mp4"))
        if (path / "poses_bounds.npy").is_file() and (has_images or has_videos):
            scenes.append(path)
        elif requested_scenes:
            raise ValueError(
                f"{path} does not contain poses_bounds.npy and either "
                "camXX.mp4 videos or cam00/images"
            )
    if not scenes:
        raise RuntimeError(f"No N3DV scenes found under {dataset_root}")
    return scenes


def load_rig(scene_path, input_source="auto"):
    poses, width, height, focal = load_n3dv_poses(scene_path)
    frame_directories = sorted(
        path for path in scene_path.glob("cam*/images") if path.is_dir()
    )
    video_paths = sorted(path for path in scene_path.glob("cam*.mp4") if path.is_file())
    complete_images = len(frame_directories) == len(poses)
    complete_videos = len(video_paths) == len(poses)
    if input_source == "images" and not complete_images:
        raise ValueError(
            f"{scene_path.name}: --input_source images requires {len(poses)} complete "
            f"camera folders, found {len(frame_directories)}"
        )
    if input_source == "videos" and not complete_videos:
        raise ValueError(
            f"{scene_path.name}: --input_source videos requires {len(poses)} camera "
            f"videos, found {len(video_paths)}"
        )
    if input_source == "auto":
        if complete_videos:
            selected_source = "videos"
        elif complete_images:
            selected_source = "images"
        else:
            raise ValueError(
                f"{scene_path.name}: neither source is complete for {len(poses)} poses "
                f"({len(frame_directories)} image folders, {len(video_paths)} videos)"
            )
    else:
        selected_source = input_source

    fps = None
    if selected_source == "images":
        if len(frame_directories) != len(poses):
            raise ValueError(
                f"{scene_path.name}: found {len(frame_directories)} camera folders "
                f"but {len(poses)} poses"
            )
        frame_paths = []
        camera_names = []
        for directory in frame_directories:
            paths = sorted(
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not paths:
                raise FileNotFoundError(f"No frames found in {directory}")
            frame_paths.append(paths)
            camera_names.append(directory.parent.name)
        frame_counts = {len(paths) for paths in frame_paths}
        if len(frame_counts) != 1:
            raise ValueError(
                f"{scene_path.name}: cameras have different frame counts: "
                f"{sorted(frame_counts)}"
            )
        frame_count = len(frame_paths[0])
        video_paths = None
        with Image.open(frame_paths[0][0]) as image:
            actual_width, actual_height = image.size
    else:
        if len(video_paths) != len(poses):
            raise ValueError(
                f"{scene_path.name}: found {len(video_paths)} camera videos "
                f"but {len(poses)} poses"
            )
        frame_paths = None
        camera_names = [path.stem for path in video_paths]
        video_metadata = []
        for video_path in video_paths:
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open N3DV video {video_path}")
                video_metadata.append((
                    int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    float(capture.get(cv2.CAP_PROP_FPS)),
                ))
            finally:
                capture.release()
        invalid_metadata = [
            (path.name, metadata)
            for path, metadata in zip(video_paths, video_metadata)
            if any(not np.isfinite(value) or value <= 0 for value in metadata)
        ]
        if invalid_metadata:
            raise ValueError(
                f"{scene_path.name}: invalid video metadata: {invalid_metadata}"
            )
        dimension_values = {metadata[:3] for metadata in video_metadata}
        fps_values = np.asarray([metadata[3] for metadata in video_metadata])
        if len(dimension_values) != 1 or not np.allclose(
            fps_values,
            fps_values[0],
            rtol=1e-5,
            atol=1e-5,
        ):
            raise ValueError(
                f"{scene_path.name}: camera videos have different frame counts or "
                f"resolutions/FPS: {video_metadata}"
            )
        frame_count, actual_width, actual_height, fps = video_metadata[0]
    if (actual_width, actual_height) != (width, height):
        scale_x = actual_width / width
        scale_y = actual_height / height
        if not np.isclose(scale_x, scale_y, rtol=1e-3):
            raise ValueError(
                f"{scene_path.name}: image size {(actual_width, actual_height)} is not a uniform "
                f"scale of poses_bounds size {(width, height)}"
            )
        focal *= scale_x
        width, height = actual_width, actual_height

    intrinsic = np.array([
        [focal, 0.0, width * 0.5],
        [0.0, focal, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    world_to_cameras = []
    projections = []
    camera_centers = []
    for pose in poses:
        camera_to_world = pose.copy()
        # poses_bounds stores OpenGL camera axes; OpenCV projection uses x-right,
        # y-down, z-forward.
        camera_to_world[:3, 1:3] *= -1
        world_to_camera = np.linalg.inv(camera_to_world)
        world_to_cameras.append(world_to_camera)
        projections.append(intrinsic @ world_to_camera[:3, :])
        camera_centers.append(camera_to_world[:3, 3])

    return N3DVRig(
        scene_path=scene_path,
        camera_names=camera_names,
        frame_paths=frame_paths,
        video_paths=video_paths,
        frame_count_value=frame_count,
        projections=np.asarray(projections),
        world_to_cameras=np.asarray(world_to_cameras),
        camera_centers=np.asarray(camera_centers),
        width=width,
        height=height,
        focal=focal,
        fps=fps,
    )


def select_camera_pairs(camera_centers, pairs_per_camera, camera_indices=None):
    if pairs_per_camera <= 0:
        raise ValueError("--pairs_per_camera must be positive")
    camera_centers = np.asarray(camera_centers)
    if camera_indices is None:
        camera_indices = list(range(len(camera_centers)))
    else:
        camera_indices = [int(index) for index in camera_indices]
    if len(camera_indices) < 2:
        raise ValueError("At least two included cameras are required for triangulation")
    pair_set = set()
    included_centers = camera_centers[camera_indices]
    neighbor_count = min(pairs_per_camera, len(camera_indices) - 1)
    for local_index, camera_index in enumerate(camera_indices):
        distances = np.linalg.norm(included_centers - camera_centers[camera_index], axis=1)
        distances[local_index] = np.inf
        neighbors = np.argsort(distances)[:neighbor_count]
        for local_neighbor in neighbors:
            neighbor = camera_indices[int(local_neighbor)]
            pair_set.add(tuple(sorted((camera_index, neighbor))))
    return sorted(pair_set)


def load_roma_model(model_name, device):
    try:
        from romatch import roma_indoor, roma_outdoor, tiny_roma_v1_outdoor
    except ImportError as error:
        raise RuntimeError(
            "RoMa is required for N3DV point-cloud generation. Install it with "
            "`pip install romatch`."
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({device}) but torch.cuda.is_available() is false")
    factory = {
        "outdoor": roma_outdoor,
        "indoor": roma_indoor,
        "tiny": tiny_roma_v1_outdoor,
    }[model_name]
    print(f"Loading RoMa model {model_name!r} on {device}; weights may download on first use")
    if model_name == "tiny":
        return factory(device=device)

    # Full RoMa enables its optional fused kernel by default on Linux. Wheels
    # for that extension are tied to specific PyTorch releases, so silently
    # installing a mismatched wheel can replace or break the project's CUDA
    # stack. RoMa provides an equivalent native-PyTorch implementation.
    try:
        import local_corr  # noqa: F401
    except (ImportError, OSError) as error:
        print(
            "  fused local_corr is unavailable "
            f"({type(error).__name__}: {error}); using RoMa's native PyTorch correlation"
        )
        print("  This is slower but does not change the preprocessing format.")
        return factory(device=device, use_custom_corr=False)
    return factory(device=device, use_custom_corr=True)


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sample_roma_matches(model, image_a, image_b, width, height, args):
    with torch.inference_mode():
        if args.roma_model == "tiny":
            warp, certainty = model.match(image_a, image_b)
        else:
            warp, certainty = model.match(image_a, image_b, device=args.device)
        try:
            matches, match_certainty = model.sample(
                warp,
                certainty,
                num=args.matches_per_pair,
            )
        except TypeError:
            matches, match_certainty = model.sample(warp, certainty)
        keypoints_a, keypoints_b = model.to_pixel_coordinates(
            matches,
            height,
            width,
            height,
            width,
        )

    keypoints_a = tensor_to_numpy(keypoints_a).astype(np.float64)
    keypoints_b = tensor_to_numpy(keypoints_b).astype(np.float64)
    match_certainty = tensor_to_numpy(match_certainty).reshape(-1).astype(np.float64)
    finite = (
        np.isfinite(keypoints_a).all(axis=1)
        & np.isfinite(keypoints_b).all(axis=1)
        & np.isfinite(match_certainty)
        & (match_certainty >= args.min_certainty)
    )
    keypoints_a = keypoints_a[finite]
    keypoints_b = keypoints_b[finite]
    match_certainty = match_certainty[finite]

    if len(keypoints_a) > args.matches_per_pair:
        best = np.argpartition(match_certainty, -args.matches_per_pair)[-args.matches_per_pair:]
        keypoints_a = keypoints_a[best]
        keypoints_b = keypoints_b[best]
        match_certainty = match_certainty[best]
    if len(keypoints_a) < 8:
        return keypoints_a, keypoints_b, match_certainty

    _, inliers = cv2.findFundamentalMat(
        keypoints_a,
        keypoints_b,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=args.ransac_threshold,
        confidence=0.999,
        maxIters=10_000,
    )
    if inliers is not None:
        inliers = inliers.reshape(-1).astype(bool)
        keypoints_a = keypoints_a[inliers]
        keypoints_b = keypoints_b[inliers]
        match_certainty = match_certainty[inliers]
    return keypoints_a, keypoints_b, match_certainty


def triangulate_matches(
    keypoints_a,
    keypoints_b,
    projection_a,
    projection_b,
    world_to_camera_a,
    world_to_camera_b,
    max_reprojection_error,
    max_scene_radius,
):
    if len(keypoints_a) < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        keypoints_a.T,
        keypoints_b.T,
    ).T
    valid_w = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 3]) > 1e-10)
    points = np.zeros((len(homogeneous), 3), dtype=np.float64)
    points[valid_w] = homogeneous[valid_w, :3] / homogeneous[valid_w, 3:4]
    points_h = np.concatenate([points, np.ones((len(points), 1))], axis=1)

    camera_a = points_h @ world_to_camera_a.T
    camera_b = points_h @ world_to_camera_b.T
    projected_a_h = points_h @ projection_a.T
    projected_b_h = points_h @ projection_b.T
    valid_depth = (
        (camera_a[:, 2] > 1e-6)
        & (camera_b[:, 2] > 1e-6)
        & (projected_a_h[:, 2] > 1e-10)
        & (projected_b_h[:, 2] > 1e-10)
    )
    projected_a = projected_a_h[:, :2] / np.maximum(projected_a_h[:, 2:3], 1e-10)
    projected_b = projected_b_h[:, :2] / np.maximum(projected_b_h[:, 2:3], 1e-10)
    reprojection_error = np.maximum(
        np.linalg.norm(projected_a - keypoints_a, axis=1),
        np.linalg.norm(projected_b - keypoints_b, axis=1),
    )
    valid = (
        valid_w
        & valid_depth
        & np.isfinite(points).all(axis=1)
        & (reprojection_error <= max_reprojection_error)
        & (np.linalg.norm(points, axis=1) <= max_scene_radius)
    )
    return points.astype(np.float32), valid


def sample_image_colors(image_rgb, keypoints):
    x = np.clip(np.rint(keypoints[:, 0]).astype(np.int64), 0, image_rgb.shape[1] - 1)
    y = np.clip(np.rint(keypoints[:, 1]).astype(np.int64), 0, image_rgb.shape[0] - 1)
    return image_rgb[y, x].astype(np.float32) / 255.0


def confidence_voxel_filter(points, colors, confidence, voxel_size, max_points):
    if len(points) == 0:
        return points, colors, confidence
    selected = np.arange(len(points))
    if voxel_size > 0:
        voxels = np.floor(points / voxel_size).astype(np.int64)
        confidence_order = np.argsort(-confidence)
        _, first = np.unique(voxels[confidence_order], axis=0, return_index=True)
        selected = confidence_order[first]
    if max_points > 0 and len(selected) > max_points:
        best = np.argpartition(confidence[selected], -max_points)[-max_points:]
        selected = selected[best]
    return points[selected], colors[selected], confidence[selected]


def write_point_cloud_ply(path, points, colors):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(len(points), dtype=dtype)
    normals = np.zeros_like(points, dtype=np.float32)
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    attributes = np.concatenate([points.astype(np.float32), normals, rgb], axis=1)
    vertices[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def _selected_blender_frame_paths(rig, frame_start, frame_end):
    """Return materialized source images for the selected source-frame range."""
    selected_count = int(frame_end) - int(frame_start) + 1
    if rig.extracted_frame_paths is not None:
        if rig.extracted_frame_start != int(frame_start):
            raise ValueError(
                "Extracted N3DV images do not start at the selected Blender clip: "
                f"images={rig.extracted_frame_start}, clip={frame_start}"
            )
        paths = rig.extracted_frame_paths
    elif rig.frame_paths is not None:
        paths = [
            camera_paths[int(frame_start):int(frame_end) + 1]
            for camera_paths in rig.frame_paths
        ]
    else:
        raise RuntimeError(
            "--blender_output requires materialized images. Keep --extract_images "
            "enabled when preprocessing camXX.mp4 inputs."
        )
    if len(paths) != len(rig.camera_names) or any(
            len(camera_paths) != selected_count for camera_paths in paths):
        raise ValueError(
            "Selected Blender image sources do not match the camera/clip dimensions"
        )
    return paths


def _images_have_same_rgb_pixels(first_path, second_path):
    try:
        with Image.open(first_path) as first_image, Image.open(second_path) as second_image:
            if first_image.size != second_image.size:
                return False
            return np.array_equal(
                np.asarray(first_image.convert("RGB")),
                np.asarray(second_image.convert("RGB")),
            )
    except (OSError, ValueError):
        return False


def _materialize_blender_png(source_path, destination, overwrite=False):
    """Copy/convert one selected frame into the flat legacy images directory."""
    source_path = Path(source_path)
    destination = Path(destination)
    destination_exists = os.path.lexists(destination)
    if destination_exists and _images_have_same_rgb_pixels(source_path, destination):
        return False
    if destination_exists and not overwrite:
        raise FileExistsError(
            f"Refusing to replace conflicting Blender image {destination}; "
            "pass --overwrite to rebuild legacy outputs"
        )
    if destination.is_dir():
        raise FileExistsError(
            f"Cannot replace Blender image because it is a directory: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.png")
    try:
        if source_path.suffix.lower() == ".png":
            shutil.copy2(source_path, temporary)
        else:
            with Image.open(source_path) as source_image:
                source_image.convert("RGB").save(temporary, format="PNG")
        if not _images_have_same_rgb_pixels(source_path, temporary):
            raise RuntimeError(
                f"Materialized Blender image does not match its source: {destination}"
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def _write_json_if_compatible(path, payload, overwrite=False):
    """Atomically write deterministic JSON without replacing unrelated outputs."""
    path = Path(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.is_file() and path.read_bytes() == encoded:
        return False
    if os.path.lexists(path) and not overwrite:
        raise FileExistsError(
            f"Refusing to replace conflicting Blender metadata {path}; "
            "pass --overwrite to rebuild legacy outputs"
        )
    if path.is_dir():
        raise FileExistsError(
            f"Cannot replace Blender metadata because it is a directory: {path}"
        )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "wb") as output_file:
            output_file.write(encoded)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return True


def build_blender_outputs(
    rig,
    frame_start,
    frame_end,
    output_dir,
    reconstruction_context,
    overwrite=False,
):
    """Create legacy Blender/OMG4 files from the dynamic preprocessing results.

    The flat image set reuses already extracted/native frames, and points3d.ply
    uses the RoMa reconstruction at the first selected time. No second FFmpeg or
    COLMAP pass is needed.
    """
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    selected_count = frame_end - frame_start + 1
    frame_paths = _selected_blender_frame_paths(rig, frame_start, frame_end)
    poses, _, _, _ = load_n3dv_poses(rig.scene_path)
    if len(poses) != len(rig.camera_names):
        raise ValueError(
            f"N3DV has {len(rig.camera_names)} camera sources but {len(poses)} poses"
        )

    flat_images_dir = rig.scene_path / "images"
    train_frames = []
    test_frames = []
    written_images = 0
    for camera_index, (camera_name, camera_paths) in enumerate(zip(
        rig.camera_names,
        frame_paths,
    )):
        destination_frames = test_frames if camera_index == 0 else train_frames
        for local_index, source_path in enumerate(camera_paths):
            source_frame_index = frame_start + local_index
            image_name = f"{camera_name}_{source_frame_index:04d}.png"
            destination = flat_images_dir / image_name
            written_images += int(_materialize_blender_png(
                source_path,
                destination,
                overwrite=overwrite,
            ))
            destination_frames.append({
                "file_path": f"images/{Path(image_name).stem}",
                "transform_matrix": poses[camera_index].tolist(),
                "time": local_index / selected_count,
            })

    common_metadata = {
        "w": int(rig.width),
        "h": int(rig.height),
        "fl_x": float(rig.focal),
        "fl_y": float(rig.focal),
        "cx": float(rig.width) * 0.5,
        "cy": float(rig.height) * 0.5,
    }
    train_transforms = dict(common_metadata, frames=train_frames)
    test_transforms = dict(common_metadata, frames=test_frames)
    _write_json_if_compatible(
        rig.scene_path / "transforms_train.json",
        train_transforms,
        overwrite=overwrite,
    )
    _write_json_if_compatible(
        rig.scene_path / "transforms_test.json",
        test_transforms,
        overwrite=overwrite,
    )

    points_path, colors_path, _ = frame_output_paths(output_dir, frame_start)
    points, colors = _load_frame_arrays(points_path, colors_path)
    points_output = rig.scene_path / "points3d.ply"
    manifest_path = output_dir / BLENDER_OUTPUT_MANIFEST_NAME
    source_fingerprint = {
        "points": _file_signature(points_path, hash_contents=True),
        "colors": _file_signature(colors_path, hash_contents=True),
    }
    manifest_base = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "scene_name": rig.scene_path.name,
        "scene_fingerprint": reconstruction_context["scene"]["fingerprint"],
        "reconstruction_fingerprint": reconstruction_context["fingerprint"],
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": selected_count,
        "camera_names": list(rig.camera_names),
        "held_out_camera": rig.camera_names[0],
        "train_frame_count": len(train_frames),
        "test_frame_count": len(test_frames),
        "point_cloud_source_frame": frame_start,
        "point_cloud_source_fingerprint": source_fingerprint,
    }
    published = _read_self_fingerprinted_manifest(manifest_path)
    published_payload = published[1] if published is not None else None
    previous_point_signature = (
        published_payload.get("points3d_signature")
        if published_payload is not None else None
    )
    current_point_signature = (
        _file_signature(points_output, hash_contents=True)
        if points_output.is_file() else None
    )
    previous_base_matches = (
        published_payload is not None
        and all(published_payload.get(key) == value for key, value in manifest_base.items())
    )
    point_cloud_is_current = (
        previous_base_matches
        and previous_point_signature == current_point_signature
    )
    if not point_cloud_is_current:
        if os.path.lexists(points_output) and not overwrite:
            raise FileExistsError(
                f"Refusing to replace unverified Blender point cloud {points_output}; "
                "pass --overwrite to rebuild legacy outputs"
            )
        temporary_ply = points_output.with_name(
            f".{points_output.name}.{os.getpid()}.tmp.ply"
        )
        try:
            write_point_cloud_ply(temporary_ply, points, colors)
            os.replace(temporary_ply, points_output)
        finally:
            if temporary_ply.exists():
                temporary_ply.unlink()

    manifest_payload = dict(manifest_base)
    manifest_payload["points3d_signature"] = _file_signature(
        points_output,
        hash_contents=True,
    )
    manifest = _publish_extracted_images_manifest(manifest_path, manifest_payload)
    retained_images = len(rig.camera_names) * selected_count - written_images
    print(
        f"  Blender/OMG4 output: {len(train_frames)} train + "
        f"{len(test_frames)} test frames; images wrote={written_images}, "
        f"retained={retained_images}; point cloud={points_output}; "
        f"manifest={manifest_path} ({manifest['fingerprint'][:12]})"
    )


def frame_output_paths(output_dir, frame_index):
    return (
        output_dir / f"points3d_frame{frame_index:06d}.npy",
        output_dir / f"colors_frame{frame_index:06d}.npy",
        output_dir / "point_clouds" / f"frame{frame_index:06d}.ply",
    )


def frame_cache_metadata_path(output_dir, frame_index):
    return output_dir / f"points3d_frame{frame_index:06d}{CACHE_METADATA_SUFFIX}"


def _load_frame_arrays(points_path, colors_path):
    points = np.asarray(np.load(points_path), dtype=np.float32)
    colors = np.asarray(np.load(colors_path), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"Cached points must be a non-empty Nx3 array, got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(
            f"Cached colors must have the same Nx3 shape as points, got {colors.shape}"
        )
    if not np.isfinite(points).all() or not np.isfinite(colors).all():
        raise ValueError("Cached frame arrays contain non-finite values")
    return points, colors


def inspect_frame_cache(
    rig,
    frame_index,
    output_dir,
    camera_indices,
    reconstruction_context,
    allow_legacy=False,
):
    points_path, colors_path, _ = frame_output_paths(output_dir, frame_index)
    metadata_path = frame_cache_metadata_path(output_dir, frame_index)
    if not points_path.is_file() or not colors_path.is_file():
        return None, None, "missing", "point/color arrays are missing"
    try:
        points, colors = _load_frame_arrays(points_path, colors_path)
    except (OSError, ValueError) as error:
        return None, None, "invalid", str(error)
    expected = frame_cache_provenance(
        rig,
        frame_index,
        camera_indices,
        reconstruction_context,
    )
    if not metadata_path.is_file():
        if allow_legacy:
            return points, colors, "legacy_unverified", "provenance sidecar is missing"
        return None, None, "unverified", "provenance sidecar is missing"
    try:
        with open(metadata_path, encoding="utf-8") as metadata_file:
            actual = json.load(metadata_file)
    except (OSError, ValueError, TypeError) as error:
        return None, None, "invalid", f"could not read provenance sidecar: {error}"
    if actual.get("fingerprint") != expected["fingerprint"]:
        return None, None, "mismatch", (
            "provenance fingerprint differs (scene, source files, cameras, settings, "
            "or seed changed)"
        )
    return points, colors, "verified", ""


def _extracted_images_manifest_payload(rig, frame_start, frame_end):
    if rig.video_paths is None:
        raise ValueError("An extracted-images manifest requires video origin sources")
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    return {
        "schema_version": EXTRACTED_IMAGES_SCHEMA_VERSION,
        "scene_name": rig.scene_path.name,
        "scene_fingerprint": build_scene_identity(rig)["fingerprint"],
        "source_kind": "videos",
        "source_camera_names": list(rig.camera_names),
        "source_frame_count": int(rig.frame_count),
        "source_fps": float(rig.fps),
        "source_width": int(rig.width),
        "source_height": int(rig.height),
        "source_frame_start": frame_start,
        "source_frame_end": frame_end,
        "extracted_frame_count": frame_end - frame_start + 1,
        "output_pattern": EXTRACTED_IMAGES_OUTPUT_PATTERN,
        "source_video_signatures": [
            _file_signature(video_path) for video_path in rig.video_paths
        ],
    }


def _read_self_fingerprinted_manifest(manifest_path):
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(manifest, dict):
        return None
    fingerprint = manifest.get("fingerprint")
    payload = {key: value for key, value in manifest.items() if key != "fingerprint"}
    if fingerprint != _json_fingerprint(payload):
        return None
    return manifest, payload


def _manifest_matches_extraction_source(payload, expected_payload):
    """Compare clip/source identity while permitting a v1-to-v2 schema upgrade."""
    return all(
        payload.get(key) == value
        for key, value in expected_payload.items()
        if key != "schema_version"
    )


def _output_image_signatures(rig, extracted_paths):
    signatures = []
    for camera_paths in extracted_paths:
        for path in camera_paths:
            path = Path(path)
            if not path.is_file():
                return None
            stat = path.stat()
            signatures.append({
                "path": path.relative_to(rig.scene_path).as_posix(),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
    return signatures


def _invalidate_published_extracted_images_manifest(manifest_path, manifest=None):
    """Atomically hide a published manifest before changing its bound outputs."""
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
    if not isinstance(fingerprint, str) or not fingerprint:
        digest = hashlib.sha256()
        with open(manifest_path, "rb") as manifest_file:
            for chunk in iter(lambda: manifest_file.read(1024 * 1024), b""):
                digest.update(chunk)
        fingerprint = digest.hexdigest()
    backup_path = manifest_path.with_name(
        f"{manifest_path.stem}.previous.{fingerprint[:16]}{manifest_path.suffix}"
    )
    os.replace(manifest_path, backup_path)
    print(f"  extracted images: unpublished old manifest -> {backup_path.name}")
    return backup_path


def _publish_extracted_images_manifest(manifest_path, payload):
    manifest = dict(payload)
    manifest["fingerprint"] = _json_fingerprint(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    with open(temporary_manifest, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
    os.replace(temporary_manifest, manifest_path)
    return manifest


def _read_valid_extracted_png(path, width, height):
    if not Path(path).is_file():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if (
        image is None
        or image.dtype != np.uint8
        or image.ndim != 3
        or image.shape != (int(height), int(width), 3)
    ):
        return None
    return image


def _has_valid_extracted_png_header(path, width, height):
    """Cheaply validate a published PNG without decoding all image pixels."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            return (
                image.format == "PNG"
                and image.mode == "RGB"
                and image.size == (int(width), int(height))
            )
    except (OSError, ValueError):
        return False


def _selected_extracted_frame_paths(rig, frame_start, frame_end):
    frame_count = int(frame_end) - int(frame_start) + 1
    return [
        [
            rig.scene_path / camera_name / "images" / f"{local_frame:04d}.png"
            for local_frame in range(frame_count)
        ]
        for camera_name in rig.camera_names
    ]


def extract_selected_video_frames(
    rig,
    frame_start,
    frame_end,
    output_dir,
    overwrite=False,
):
    """Sequentially materialize a selected video clip as clip-local PNG frames.

    Unrelated files are never removed, and existing canonical frames are only
    replaced with explicit ``overwrite`` authorization. A complete,
    self-fingerprinted manifest makes subsequent calls idempotent; interrupted
    legacy extractions resume by comparing decoded frames with their PNG targets.
    """
    if rig.video_paths is None:
        raise ValueError("Frame extraction is only valid for an N3DV video rig")
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    if frame_start < 0 or frame_end < frame_start or frame_end >= rig.frame_count:
        raise ValueError(
            f"Invalid extraction range [{frame_start}, {frame_end}] for "
            f"{rig.frame_count} source frames"
        )

    output_dir = Path(output_dir)
    manifest_path = output_dir / EXTRACTED_IMAGES_MANIFEST_NAME
    expected_payload = _extracted_images_manifest_payload(rig, frame_start, frame_end)
    extracted_paths = _selected_extracted_frame_paths(rig, frame_start, frame_end)
    published = _read_self_fingerprinted_manifest(manifest_path)
    published_manifest = published[0] if published is not None else None
    published_payload = published[1] if published is not None else None
    if (
        published_payload is not None
        and not _manifest_matches_extraction_source(published_payload, expected_payload)
        and not overwrite
    ):
        raise FileExistsError(
            f"{manifest_path} publishes a different clip or video source "
            f"(frames {published_payload.get('source_frame_start')}.."
            f"{published_payload.get('source_frame_end')}); pass --overwrite to "
            f"replace it with frames {frame_start}..{frame_end}"
        )

    output_signatures = _output_image_signatures(rig, extracted_paths)
    all_headers_valid = output_signatures is not None and all(
        _has_valid_extracted_png_header(path, rig.width, rig.height)
        for camera_paths in extracted_paths
        for path in camera_paths
    )
    expected_v2_payload = dict(expected_payload)
    expected_v2_payload["output_image_signatures"] = output_signatures
    if (
        not overwrite
        and all_headers_valid
        and published_payload == expected_v2_payload
    ):
        rig.extracted_frame_paths = extracted_paths
        rig.extracted_frame_start = frame_start
        print(
            f"  extracted images: verified {len(rig.camera_names)} cameras x "
            f"{expected_payload['extracted_frame_count']} frames and output signatures"
        )
        return published_manifest

    # Safely upgrade the exact v1 manifest emitted by the previous extractor.
    # Its self-fingerprint asserted a completed same-source clip; binding the
    # current header-valid files by stat does not require decoding videos again.
    expected_v1_payload = dict(expected_payload)
    expected_v1_payload["schema_version"] = 1
    if (
        not overwrite
        and all_headers_valid
        and published_payload == expected_v1_payload
    ):
        _invalidate_published_extracted_images_manifest(
            manifest_path,
            published_manifest,
        )
        manifest = _publish_extracted_images_manifest(
            manifest_path,
            expected_v2_payload,
        )
        rig.extracted_frame_paths = extracted_paths
        rig.extracted_frame_start = frame_start
        print(
            f"  extracted images: upgraded same-clip manifest v1 -> v2 "
            f"({manifest['fingerprint'][:12]})"
        )
        return manifest

    manifest_unpublished = False

    def unpublish_before_mutation():
        nonlocal manifest_unpublished
        if manifest_unpublished:
            return
        _invalidate_published_extracted_images_manifest(
            manifest_path,
            published_manifest,
        )
        manifest_unpublished = True

    extracted_count = expected_payload["extracted_frame_count"]
    written_count = 0
    retained_count = 0
    for camera_index, (camera_name, video_path, camera_paths) in enumerate(zip(
        rig.camera_names,
        rig.video_paths,
        extracted_paths,
    )):
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Could not open N3DV video {video_path}")
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_start):
                raise RuntimeError(
                    f"Could not seek {video_path} to source frame {frame_start}"
                )
            for local_frame, (source_frame, destination) in enumerate(zip(
                range(frame_start, frame_end + 1),
                camera_paths,
            )):
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    raise RuntimeError(
                        f"Could not decode source frame {source_frame} from {video_path}"
                    )
                if (
                    frame_bgr.dtype != np.uint8
                    or frame_bgr.shape != (rig.height, rig.width, 3)
                ):
                    raise RuntimeError(
                        f"Decoded {video_path.name} frame {source_frame} has shape/dtype "
                        f"{frame_bgr.shape}/{frame_bgr.dtype}; expected "
                        f"{(rig.height, rig.width, 3)}/uint8"
                    )
                destination_exists = os.path.lexists(destination)
                if not overwrite:
                    existing = _read_valid_extracted_png(
                        destination,
                        rig.width,
                        rig.height,
                    )
                    if existing is not None and np.array_equal(existing, frame_bgr):
                        retained_count += 1
                        continue
                    if destination_exists:
                        raise FileExistsError(
                            f"Refusing to overwrite existing canonical extracted image "
                            f"{destination}: it is invalid or does not match source frame "
                            f"{source_frame}. Pass --overwrite to replace conflicting files."
                        )
                elif destination.is_dir():
                    raise FileExistsError(
                        f"Cannot replace extracted-image path because it is a directory: "
                        f"{destination}"
                    )
                unpublish_before_mutation()
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.name}.tmp.png")
                if not cv2.imwrite(
                    str(temporary),
                    frame_bgr,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3],
                ):
                    raise RuntimeError(f"Could not write extracted PNG {temporary}")
                os.replace(temporary, destination)
                written_count += 1
        finally:
            capture.release()
        print(
            f"  extracted {camera_name}: {extracted_count} frames "
            f"({camera_index + 1}/{len(rig.camera_names)})"
        )

    # Decode every expected PNG once before publishing the manifest. A failed
    # or partial run therefore cannot be mistaken for a complete image cache.
    invalid_outputs = [
        path
        for camera_paths in extracted_paths
        for path in camera_paths
        if not _has_valid_extracted_png_header(path, rig.width, rig.height)
    ]
    if invalid_outputs:
        raise RuntimeError(
            f"Extracted image validation failed for {len(invalid_outputs)} files; "
            f"first invalid output: {invalid_outputs[0]}"
        )

    output_signatures = _output_image_signatures(rig, extracted_paths)
    if output_signatures is None:
        raise RuntimeError("Extracted image signatures could not be collected")
    manifest_payload = dict(expected_payload)
    manifest_payload["output_image_signatures"] = output_signatures
    unpublish_before_mutation()
    manifest = _publish_extracted_images_manifest(manifest_path, manifest_payload)
    rig.extracted_frame_paths = extracted_paths
    rig.extracted_frame_start = frame_start
    print(
        f"  extracted images complete: wrote {written_count:,}, retained {retained_count:,}; "
        f"manifest={manifest_path} ({manifest['fingerprint'][:12]})"
    )
    return manifest


def decode_video_frame(video_path, frame_index):
    capture = cv2.VideoCapture(str(video_path))
    success = False
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open N3DV video {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame_bgr = capture.read()
    finally:
        capture.release()
    if not success:
        raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def reconstruct_frame(
    rig,
    frame_index,
    camera_pairs,
    model,
    args,
    output_dir,
    camera_indices=None,
    reconstruction_context=None,
):
    points_path, colors_path, ply_path = frame_output_paths(output_dir, frame_index)
    if camera_indices is None:
        camera_indices = sorted({index for pair in camera_pairs for index in pair})
    if reconstruction_context is None:
        reconstruction_context = build_reconstruction_context(
            rig,
            camera_pairs,
            camera_indices,
            [],
            args,
        )
    expected_provenance = frame_cache_provenance(
        rig,
        frame_index,
        camera_indices,
        reconstruction_context,
    )
    if not args.overwrite:
        points, colors, cache_status, cache_reason = inspect_frame_cache(
            rig,
            frame_index,
            output_dir,
            camera_indices,
            reconstruction_context,
        )
        if cache_status == "verified":
            if not args.no_ply and not ply_path.is_file():
                write_point_cloud_ply(ply_path, points, colors)
            print(f"  frame {frame_index:04d}: verified cache, {len(points):,} points")
            return points, colors
        if points_path.exists() or colors_path.exists():
            print(f"  frame {frame_index:04d}: rebuilding stale cache ({cache_reason})")

    all_points = []
    all_colors = []
    all_confidence = []
    image_cache = {}
    max_scene_radius = np.median(np.linalg.norm(rig.camera_centers, axis=1)) * args.scene_radius_factor
    for camera_a, camera_b in tqdm(
        camera_pairs,
        desc=f"  frame {frame_index:04d} camera pairs",
        leave=False,
    ):
        pair_seed = derived_seed(
            args.seed,
            reconstruction_context["scene"]["fingerprint"],
            frame_index,
            camera_a,
            camera_b,
        )
        seed_random_generators(pair_seed)
        if rig.extracted_frame_paths is not None:
            local_frame = frame_index - int(rig.extracted_frame_start)
            image_a = rig.extracted_frame_paths[camera_a][local_frame]
            image_b = rig.extracted_frame_paths[camera_b][local_frame]
        elif rig.video_paths is not None:
            for camera_index in (camera_a, camera_b):
                if camera_index not in image_cache:
                    image_cache[camera_index] = decode_video_frame(
                        rig.video_paths[camera_index], frame_index
                    )
            image_a = image_cache[camera_a]
            image_b = image_cache[camera_b]
        else:
            image_a = rig.frame_paths[camera_a][frame_index]
            image_b = rig.frame_paths[camera_b][frame_index]
        keypoints_a, keypoints_b, confidence = sample_roma_matches(
            model,
            image_a,
            image_b,
            rig.width,
            rig.height,
            args,
        )
        points, valid = triangulate_matches(
            keypoints_a,
            keypoints_b,
            rig.projections[camera_a],
            rig.projections[camera_b],
            rig.world_to_cameras[camera_a],
            rig.world_to_cameras[camera_b],
            args.max_reprojection_error,
            max_scene_radius,
        )
        if not np.any(valid):
            continue
        if rig.extracted_frame_paths is not None:
            if camera_a not in image_cache:
                image_bgr = cv2.imread(str(image_a), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Could not read {image_a}")
                image_cache[camera_a] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = image_cache[camera_a]
        elif rig.video_paths is not None:
            image_rgb = np.asarray(image_cache[camera_a])
        else:
            if camera_a not in image_cache:
                image_bgr = cv2.imread(str(image_a), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RuntimeError(f"Could not read {image_a}")
                image_cache[camera_a] = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb = image_cache[camera_a]
        colors = sample_image_colors(image_rgb, keypoints_a)
        all_points.append(points[valid])
        all_colors.append(colors[valid])
        all_confidence.append(confidence[valid].astype(np.float32))

    if not all_points:
        raise RuntimeError(
            f"{rig.scene_path.name} frame {frame_index}: no valid points triangulated. "
            "Try lowering --min_certainty or increasing --max_reprojection_error."
        )
    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    confidence = np.concatenate(all_confidence)
    points, colors, _ = confidence_voxel_filter(
        points,
        colors,
        confidence,
        args.voxel_size,
        args.max_points_per_frame,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(points_path, points.astype(np.float32))
    np.save(colors_path, colors.astype(np.float32))
    with open(frame_cache_metadata_path(output_dir, frame_index), "w", encoding="utf-8") as metadata_file:
        json.dump(expected_provenance, metadata_file, indent=2, sort_keys=True)
    if not args.no_ply:
        write_point_cloud_ply(ply_path, points, colors)
    print(f"  frame {frame_index:04d}: triangulated {len(points):,} points")
    return points, colors


def stratified_point_budget(frame_arrays, max_points, rng):
    total_points = sum(len(points) for points in frame_arrays)
    if max_points <= 0 or total_points <= max_points:
        return [np.arange(len(points)) for points in frame_arrays]
    capacities = np.asarray([len(points) for points in frame_arrays], dtype=np.int64)
    counts = np.zeros(len(frame_arrays), dtype=np.int64)
    remaining_budget = int(max_points)
    while remaining_budget > 0:
        active = np.flatnonzero(counts < capacities)
        if len(active) == 0:
            break
        if remaining_budget < len(active):
            # When there are fewer slots than frames, distribute them across
            # the time range instead of exceeding the budget with quota=1.
            offsets = ((np.arange(remaining_budget) + 0.5) * len(active) / remaining_budget)
            chosen = active[np.floor(offsets).astype(np.int64)]
            counts[chosen] += 1
            remaining_budget = 0
            break
        fair_share = max(remaining_budget // len(active), 1)
        increments = np.minimum(capacities[active] - counts[active], fair_share)
        counts[active] += increments
        remaining_budget -= int(increments.sum())

    return [
        rng.choice(len(points), int(count), replace=False)
        if count > 0 else np.empty(0, dtype=np.int64)
        for points, count in zip(frame_arrays, counts)
    ]


def resolve_velocity_distance(rig, args, selected_frame_count, included_camera_names=None):
    if included_camera_names is None:
        included_indices = list(range(len(rig.camera_names)))
    else:
        included = set(included_camera_names)
        included_indices = [
            index for index, name in enumerate(rig.camera_names) if name in included
        ]
    scene_scale = centered_camera_rig_radius(rig.camera_centers[included_indices])
    scene_limit = scene_scale * float(getattr(args, "velocity_distance_scene_fraction", 0.05))
    normalized_cap = float(getattr(args, "normalized_velocity_cap", 0.0))
    cap_aligned_limit = (
        normalized_cap / selected_frame_count if normalized_cap > 0 else 0.0
    )
    requested = getattr(args, "max_velocity_distance", None)
    if requested is None:
        if normalized_cap > 0:
            effective = min(scene_limit, cap_aligned_limit)
            method = "min_scene_scale_and_normalized_cap"
        else:
            effective = scene_limit
            method = "scene_scale"
    else:
        effective = float(requested)
        method = "explicit"
    return {
        "effective": float(effective),
        "method": method,
        "scene_scale": float(scene_scale),
        "scene_scaled_limit": float(scene_limit),
        "normalized_cap": float(normalized_cap),
        "cap_aligned_per_frame_limit": float(cap_aligned_limit),
    }


def nearest_frame_correspondences(positions, next_positions, match_mode):
    distances, indices = cKDTree(next_positions).query(positions, k=1, workers=-1)
    finite = np.isfinite(distances)
    reciprocal = np.zeros(len(positions), dtype=bool)
    if np.any(finite):
        _, reverse_indices = cKDTree(positions).query(next_positions, k=1, workers=-1)
        source_indices = np.arange(len(positions), dtype=np.int64)
        reciprocal[finite] = reverse_indices[indices[finite]] == source_indices[finite]
    if match_mode == "mutual":
        correspondence_filter = reciprocal
    elif match_mode == "forward":
        correspondence_filter = finite
    else:
        raise ValueError(f"Unsupported velocity match mode: {match_mode}")
    return distances, indices, finite, correspondence_filter, reciprocal


def build_velocity_initialization(
    rig,
    keyframes,
    output_dir,
    args,
    selected_frame_end=None,
    provenance=None,
    included_camera_names=None,
    excluded_camera_names=None,
    cache_provenance_status="unverified",
    extracted_images_manifest=None,
):
    selected_frame_start = int(args.frame_start)
    if selected_frame_end is None:
        selected_frame_end = min(int(args.frame_end), rig.frame_count - 1)
    selected_frame_end = int(selected_frame_end)
    selected_frame_count = selected_frame_end - selected_frame_start + 1
    if selected_frame_count <= 1:
        raise ValueError("Velocity initialization requires at least two selected frames")
    if any(
        frame_index < selected_frame_start or frame_index + 1 > selected_frame_end
        for frame_index in keyframes
    ):
        raise ValueError("Every velocity keyframe and its following frame must be in the selected clip")
    if included_camera_names is None:
        included_camera_names = list(rig.camera_names)
    if excluded_camera_names is None:
        excluded_camera_names = []

    threshold = resolve_velocity_distance(
        rig,
        args,
        selected_frame_count,
        included_camera_names,
    )
    match_mode = getattr(args, "velocity_match_mode", "mutual")
    positions_by_frame = []
    colors_by_frame = []
    velocities_by_frame = []
    valid_by_frame = []
    forward_match_count = 0
    reciprocal_match_count = 0
    within_distance_count = 0
    for frame_index in keyframes:
        points_path, colors_path, _ = frame_output_paths(output_dir, frame_index)
        next_points_path, _, _ = frame_output_paths(output_dir, frame_index + 1)
        if not next_points_path.is_file():
            raise FileNotFoundError(
                f"Missing following-frame point cache for frame {frame_index}: {next_points_path}"
            )
        positions, colors = _load_frame_arrays(points_path, colors_path)
        velocities = np.zeros_like(positions)
        valid = np.zeros(len(positions), dtype=bool)
        next_positions = np.asarray(np.load(next_points_path), dtype=np.float32)
        if next_positions.ndim != 2 or next_positions.shape[1] != 3 or not len(next_positions):
            raise ValueError(f"Following-frame points must be a non-empty Nx3 array: {next_points_path}")
        if not np.isfinite(next_positions).all():
            raise ValueError(f"Following-frame points contain non-finite values: {next_points_path}")
        distances, indices, finite, correspondence_filter, reciprocal = nearest_frame_correspondences(
            positions,
            next_positions,
            match_mode,
        )
        within_distance = finite & (distances < threshold["effective"])
        valid = within_distance & correspondence_filter
        velocities[valid] = next_positions[indices[valid]] - positions[valid]
        forward_match_count += int(finite.sum())
        reciprocal_match_count += int((finite & reciprocal).sum())
        within_distance_count += int(within_distance.sum())
        positions_by_frame.append(positions)
        colors_by_frame.append(colors)
        velocities_by_frame.append(velocities)
        valid_by_frame.append(valid)

    scene_identity = build_scene_identity(rig) if provenance is None else provenance["scene"]
    sampling_seed = derived_seed(args.seed, scene_identity["fingerprint"], "point_budget")
    rng = np.random.default_rng(sampling_seed)
    selected_by_frame = stratified_point_budget(positions_by_frame, args.max_init_points, rng)
    duration = min(3.0 * args.keyframe_step / selected_frame_count, 1.0)
    positions_all = []
    colors_all = []
    velocities_all = []
    times_all = []
    durations_all = []
    valid_all = []
    for frame_index, positions, colors, velocities, valid, selected in zip(
        keyframes,
        positions_by_frame,
        colors_by_frame,
        velocities_by_frame,
        valid_by_frame,
        selected_by_frame,
    ):
        positions_all.append(positions[selected])
        colors_all.append(colors[selected])
        velocities_all.append(velocities[selected])
        relative_time = (frame_index - selected_frame_start) / selected_frame_count
        times_all.append(np.full((len(selected), 1), relative_time, dtype=np.float32))
        durations_all.append(np.full((len(selected), 1), duration, dtype=np.float32))
        valid_all.append(valid[selected])

    positions = np.concatenate(positions_all).astype(np.float32)
    colors = np.concatenate(colors_all).astype(np.float32)
    velocities = np.concatenate(velocities_all).astype(np.float32)
    times = np.concatenate(times_all)
    durations = np.concatenate(durations_all)
    has_velocity = np.concatenate(valid_all)
    valid_speeds = np.linalg.norm(velocities[has_velocity], axis=1)
    normalized_speeds = valid_speeds * selected_frame_count
    normalized_cap = threshold["normalized_cap"]
    clip_count = (
        int(np.count_nonzero(normalized_speeds > normalized_cap))
        if normalized_cap > 0 else 0
    )
    clip_fraction = clip_count / len(normalized_speeds) if len(normalized_speeds) else 0.0
    reciprocal_fraction = (
        reciprocal_match_count / forward_match_count if forward_match_count else 0.0
    )
    held_out_camera = getattr(args, "held_out_camera", "cam00")
    uses_held_out_camera = held_out_camera in included_camera_names
    evaluation_safe = cache_provenance_status == "verified" and not uses_held_out_camera
    training_source_kind = (
        "images"
        if extracted_images_manifest is not None or rig.source_kind == "images"
        else "videos"
    )
    extracted_images_fingerprint = (
        extracted_images_manifest["fingerprint"]
        if extracted_images_manifest is not None else ""
    )
    velocity_provenance = {
        "schema_version": PREPROCESS_SCHEMA_VERSION,
        "scene_fingerprint": scene_identity["fingerprint"],
        "reconstruction_fingerprint": (
            provenance["fingerprint"] if provenance is not None else "standalone"
        ),
        "frame_start": selected_frame_start,
        "frame_end": selected_frame_end,
        "selected_frame_count": selected_frame_count,
        "keyframes": [int(frame_index) for frame_index in keyframes],
        "keyframe_step": int(args.keyframe_step),
        "max_init_points": int(args.max_init_points),
        "velocity_match_mode": match_mode,
        "max_velocity_distance": threshold["effective"],
        "max_velocity_distance_mode": threshold["method"],
        "scene_camera_rig_radius": threshold["scene_scale"],
        "normalized_velocity_cap": normalized_cap,
        "cache_provenance_status": cache_provenance_status,
        "required_cache_fingerprints": (
            provenance.get("required_cache_fingerprints", {})
            if provenance is not None else {}
        ),
        "included_cameras": list(included_camera_names),
        "excluded_cameras": list(excluded_camera_names),
        "training_source_kind": training_source_kind,
        "extracted_images_fingerprint": extracted_images_fingerprint,
    }
    preprocess_fingerprint = _json_fingerprint(velocity_provenance)
    init_path = rig.scene_path / "freetime_velocity_init.npz"
    np.savez_compressed(
        init_path,
        positions=positions,
        colors=colors,
        velocities=velocities,
        times=times,
        durations=durations,
        has_velocity=has_velocity,
        velocity_units="per_frame",
        coordinate_system="n3dv",
        preprocess_schema_version=PREPROCESS_SCHEMA_VERSION,
        scene_name=rig.scene_path.name,
        scene_fingerprint=scene_identity["fingerprint"],
        poses_bounds_sha256=scene_identity["poses_bounds"].get("sha256", ""),
        source_camera_names=np.asarray(rig.camera_names),
        preprocess_fingerprint=preprocess_fingerprint,
        source_kind=rig.source_kind,
        training_source_kind=training_source_kind,
        extracted_images_manifest=(
            f"freetime_preprocess/{EXTRACTED_IMAGES_MANIFEST_NAME}"
            if extracted_images_manifest is not None else ""
        ),
        extracted_images_fingerprint=extracted_images_fingerprint,
        extracted_images_frame_start=(
            int(extracted_images_manifest["source_frame_start"])
            if extracted_images_manifest is not None else -1
        ),
        extracted_images_frame_end=(
            int(extracted_images_manifest["source_frame_end"])
            if extracted_images_manifest is not None else -1
        ),
        extracted_images_frame_count=(
            int(extracted_images_manifest["extracted_frame_count"])
            if extracted_images_manifest is not None else 0
        ),
        extracted_images_output_pattern=(
            extracted_images_manifest["output_pattern"]
            if extracted_images_manifest is not None else ""
        ),
        source_fps=float(rig.fps) if rig.fps is not None else 0.0,
        source_width=int(rig.width),
        source_height=int(rig.height),
        source_frame_count=rig.frame_count,
        frame_start=selected_frame_start,
        frame_end=selected_frame_end,
        frame_count=selected_frame_count,
        selected_frame_count=selected_frame_count,
        time_origin_frame=selected_frame_start,
        time_normalization_frames=selected_frame_count,
        time_convention="selected_clip_zero_based",
        keyframe_step=args.keyframe_step,
        keyframe_source_indices=np.asarray(keyframes, dtype=np.int32),
        max_velocity_distance=threshold["effective"],
        max_velocity_distance_mode=threshold["method"],
        scene_camera_rig_radius=threshold["scene_scale"],
        scene_scaled_velocity_limit=threshold["scene_scaled_limit"],
        normalized_velocity_cap=normalized_cap,
        cap_aligned_per_frame_limit=threshold["cap_aligned_per_frame_limit"],
        normalized_velocity_clip_count=clip_count,
        normalized_velocity_clip_fraction=clip_fraction,
        velocity_match_mode=match_mode,
        forward_match_count=forward_match_count,
        reciprocal_match_count=reciprocal_match_count,
        reciprocal_match_fraction=reciprocal_fraction,
        within_velocity_distance_count=within_distance_count,
        included_cameras=np.asarray(included_camera_names),
        excluded_cameras=np.asarray(excluded_camera_names),
        held_out_camera=held_out_camera,
        uses_held_out_camera=uses_held_out_camera,
        evaluation_safe=evaluation_safe,
        cache_provenance_status=cache_provenance_status,
        cache_camera_provenance=(
            "verified" if cache_provenance_status == "verified" else "unknown"
        ),
        legacy_cache_trusted=cache_provenance_status == "legacy_unverified",
    )
    init_ply_path = rig.scene_path / "freetime_velocity_init.ply"
    if not args.no_ply:
        write_point_cloud_ply(init_ply_path, positions, colors)
    print(
        f"  initialization: {len(positions):,} points, "
        f"{has_velocity.sum():,} valid velocities -> {init_path}"
    )
    if len(valid_speeds):
        cap_summary = (
            f"normalized-cap clips={clip_fraction:.2%}"
            if normalized_cap > 0 else "normalized-cap disabled"
        )
        print(
            f"  per-frame speed: median={np.median(valid_speeds):.6f}, "
            f"max={valid_speeds.max():.6f}; {cap_summary}"
        )
    print(
        f"  velocity matching: mode={match_mode}, threshold={threshold['effective']:.6f} "
        f"({threshold['method']}), reciprocal={reciprocal_fraction:.2%}, "
        f"evaluation_safe={evaluation_safe}"
    )
    if clip_count:
        print(
            f"  WARNING: training cap {normalized_cap:g} would clip "
            f"{clip_count:,}/{len(normalized_speeds):,} valid initialized velocities"
        )
    return init_path


def process_scene(scene_path, model, args):
    rig = load_rig(scene_path, getattr(args, "input_source", "auto"))
    frame_end = min(args.frame_end, rig.frame_count - 1)
    if args.frame_start < 0 or args.frame_start > frame_end:
        raise ValueError(
            f"{scene_path.name}: invalid frame range [{args.frame_start}, {args.frame_end}] "
            f"for {rig.frame_count} frames"
        )
    keyframes = list(range(args.frame_start, frame_end + 1, args.keyframe_step))
    # Velocity is measured from each keyframe to the immediately following
    # frame, matching the FreeTimeGsVanilla preprocessing convention.
    velocity_frames = [frame for frame in keyframes if frame + 1 <= frame_end]
    if not velocity_frames:
        raise ValueError(
            f"{scene_path.name}: selected frame range has no frame with a following frame for velocity"
        )
    required_frames = sorted(set(velocity_frames + [frame + 1 for frame in velocity_frames]))
    include_eval_camera = bool(getattr(args, "include_eval_camera", False))
    held_out_camera = getattr(args, "held_out_camera", "cam00")
    evaluation_camera = rig.camera_names[0]
    if held_out_camera != evaluation_camera:
        raise ValueError(
            f"{scene_path.name}: --held_out_camera={held_out_camera!r}, but the N3DV "
            f"loader evaluates the first poses_bounds camera {evaluation_camera!r}. "
            "Use that camera so evaluation-safety metadata cannot exclude the wrong view."
        )
    excluded_camera_names = (
        [] if include_eval_camera or held_out_camera not in rig.camera_names
        else [held_out_camera]
    )
    included_camera_indices = [
        index
        for index, name in enumerate(rig.camera_names)
        if name not in excluded_camera_names
    ]
    included_camera_names = [rig.camera_names[index] for index in included_camera_indices]
    camera_pairs = select_camera_pairs(
        rig.camera_centers,
        args.pairs_per_camera,
        included_camera_indices,
    )
    reconstruction_context = build_reconstruction_context(
        rig,
        camera_pairs,
        included_camera_indices,
        excluded_camera_names,
        args,
    )
    output_dir = scene_path / "freetime_preprocess"
    selected_frame_count = frame_end - args.frame_start + 1
    print(
        f"\n[{scene_path.name}] {len(rig.camera_names)} cameras from {rig.source_kind}, "
        f"selected {selected_frame_count}/{rig.frame_count} frames, "
        f"{len(camera_pairs)} camera pairs, {len(required_frames)} frames to reconstruct"
    )
    print(
        f"  cameras included={included_camera_names}; excluded={excluded_camera_names or 'none'}; "
        f"scene={reconstruction_context['scene']['fingerprint'][:12]}"
    )
    if args.dry_run:
        print(f"  velocity keyframes: {velocity_frames}")
        print(f"  camera pairs: {[(rig.camera_names[a], rig.camera_names[b]) for a, b in camera_pairs]}")
        return

    extracted_images_manifest = None
    if (
        rig.video_paths is not None
        and bool(getattr(args, "blender_output", False))
        and not bool(getattr(args, "extract_images", True))
    ):
        raise ValueError("--blender_output cannot be combined with --no-extract_images for video input")
    if rig.video_paths is not None and bool(getattr(args, "extract_images", True)):
        extracted_images_manifest = extract_selected_video_frames(
            rig,
            args.frame_start,
            frame_end,
            output_dir,
            overwrite=bool(getattr(args, "overwrite", False)),
        )

    cache_provenance_status = "verified"
    if getattr(args, "build_init_only", False):
        statuses = []
        for frame_index in required_frames:
            _, _, status, reason = inspect_frame_cache(
                rig,
                frame_index,
                output_dir,
                included_camera_indices,
                reconstruction_context,
                allow_legacy=bool(getattr(args, "trust_legacy_cache", False)),
            )
            if status not in {"verified", "legacy_unverified"}:
                legacy_hint = (
                    " Add --trust_legacy_cache only if these are old caches with no sidecars."
                    if status == "unverified" else ""
                )
                raise RuntimeError(
                    f"{scene_path.name} frame {frame_index}: cannot use cached reconstruction "
                    f"({status}: {reason}). Run without --build_init_only to reconstruct it."
                    f"{legacy_hint}"
                )
            statuses.append(status)
        if "legacy_unverified" in statuses:
            cache_provenance_status = "legacy_unverified"
            print(
                "  WARNING: trusting legacy caches without scene/settings/camera provenance; "
                "the output will be marked evaluation_safe=False"
            )
        else:
            print("  all required frame caches have verified provenance; RoMa skipped")
    else:
        for frame_index in required_frames:
            reconstruct_frame(
                rig,
                frame_index,
                camera_pairs,
                model,
                args,
                output_dir,
                included_camera_indices,
                reconstruction_context,
            )
    required_cache_fingerprints = {}
    for frame_index in required_frames:
        metadata_path = frame_cache_metadata_path(output_dir, frame_index)
        if metadata_path.is_file():
            with open(metadata_path, encoding="utf-8") as metadata_file:
                required_cache_fingerprints[str(frame_index)] = json.load(metadata_file)[
                    "fingerprint"
                ]
        else:
            required_cache_fingerprints[str(frame_index)] = "legacy_unverified"
    reconstruction_context["required_cache_fingerprints"] = required_cache_fingerprints
    build_velocity_initialization(
        rig,
        velocity_frames,
        output_dir,
        args,
        selected_frame_end=frame_end,
        provenance=reconstruction_context,
        included_camera_names=included_camera_names,
        excluded_camera_names=excluded_camera_names,
        cache_provenance_status=cache_provenance_status,
        extracted_images_manifest=extracted_images_manifest,
    )
    if bool(getattr(args, "blender_output", False)):
        build_blender_outputs(
            rig,
            args.frame_start,
            frame_end,
            output_dir,
            reconstruction_context,
            overwrite=bool(getattr(args, "overwrite", False)),
        )
    metadata = {
        "preprocess_schema_version": PREPROCESS_SCHEMA_VERSION,
        "scene": scene_path.name,
        "scene_fingerprint": reconstruction_context["scene"]["fingerprint"],
        "reconstruction_fingerprint": reconstruction_context["fingerprint"],
        "source_frame_count": rig.frame_count,
        "selected_frame_start": args.frame_start,
        "selected_frame_end": frame_end,
        "selected_frame_count": selected_frame_count,
        "cameras": rig.camera_names,
        "included_cameras": included_camera_names,
        "excluded_cameras": excluded_camera_names,
        "camera_pairs": [[rig.camera_names[a], rig.camera_names[b]] for a, b in camera_pairs],
        "required_frames": required_frames,
        "keyframes": velocity_frames,
        "blender_output": bool(getattr(args, "blender_output", False)),
        "cache_provenance_status": cache_provenance_status,
        "evaluation_safe": cache_provenance_status == "verified" and not include_eval_camera,
        "training_source_kind": (
            "images"
            if extracted_images_manifest is not None or rig.source_kind == "images"
            else "videos"
        ),
        "extracted_images_manifest": (
            f"freetime_preprocess/{EXTRACTED_IMAGES_MANIFEST_NAME}"
            if extracted_images_manifest is not None else None
        ),
        "extracted_images_fingerprint": (
            extracted_images_manifest["fingerprint"]
            if extracted_images_manifest is not None else None
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, sort_keys=True)


def main():
    args = parse_args()
    if args.keyframe_step <= 0:
        raise ValueError("--keyframe_step must be positive")
    if args.matches_per_pair <= 0:
        raise ValueError("--matches_per_pair must be positive")
    if args.max_reprojection_error <= 0:
        raise ValueError("--max_reprojection_error must be positive")
    if args.max_velocity_distance is not None and args.max_velocity_distance <= 0:
        raise ValueError("--max_velocity_distance must be positive when provided")
    if args.velocity_distance_scene_fraction <= 0:
        raise ValueError("--velocity_distance_scene_fraction must be positive")
    if args.normalized_velocity_cap < 0:
        raise ValueError("--normalized_velocity_cap must be non-negative")
    if args.max_init_points < 0:
        raise ValueError("--max_init_points must be non-negative")
    if args.trust_legacy_cache and not args.build_init_only:
        raise ValueError("--trust_legacy_cache is only valid with --build_init_only")
    scenes = discover_scenes(args.dataset_root, args.scenes)
    print(f"N3DV scenes: {[scene.name for scene in scenes]}")
    seed_random_generators(args.seed, deterministic=args.deterministic)
    skip_roma = args.dry_run or args.build_init_only
    model = None if skip_roma else load_roma_model(args.roma_model, args.device)
    for scene_path in scenes:
        process_scene(scene_path, model, args)
    print("\nN3DV preprocessing complete")


if __name__ == "__main__":
    main()
