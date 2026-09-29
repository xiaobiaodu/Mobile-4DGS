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

import hashlib
import os
import sys
import tempfile
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
import cv2
from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time: float = 0.0
    lazy_load: bool = False
    video_path: str = None
    frame_index: int = None
    intrinsics: np.array = None
    distortion: np.array = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def _closest_point_between_rays(origin_a, direction_a, origin_b, direction_b):
    direction_a = direction_a / np.linalg.norm(direction_a)
    direction_b = direction_b / np.linalg.norm(direction_b)
    cross = np.cross(direction_a, direction_b)
    denominator = np.linalg.norm(cross) ** 2
    origin_delta = origin_b - origin_a
    distance_a = np.linalg.det([origin_delta, direction_b, cross]) / (denominator + 1e-10)
    distance_b = np.linalg.det([origin_delta, direction_a, cross]) / (denominator + 1e-10)
    distance_a = min(distance_a, 0)
    distance_b = min(distance_b, 0)
    point = (
        origin_a + distance_a * direction_a
        + origin_b + distance_b * direction_b
    ) * 0.5
    return point, denominator


def _rotation_between_vectors(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = np.dot(source, target)
    skew = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])
    return np.eye(3) + skew + skew @ skew * (
        (1 - cosine) / (np.linalg.norm(cross) ** 2 + 1e-10)
    )


def _load_n3dv_poses(path):
    """Apply the poses_bounds conversion used by OMG4's 4D-GS preprocessor."""
    poses_bounds = np.load(os.path.join(path, "poses_bounds.npy"))
    poses_with_intrinsics = poses_bounds[:, :15].reshape(-1, 3, 5)
    height, width, focal = poses_with_intrinsics[0, :, -1]

    poses = np.concatenate([
        poses_with_intrinsics[..., 1:2],
        poses_with_intrinsics[..., 0:1],
        -poses_with_intrinsics[..., 2:3],
        poses_with_intrinsics[..., 3:4],
    ], axis=-1)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses = np.concatenate([poses, last_row], axis=1)

    poses[:, :3, 1:3] *= -1
    poses = poses[:, [1, 0, 2, 3], :]
    poses[:, 2, :] *= -1

    up = poses[:, :3, 1].sum(axis=0)
    rotation = np.pad(_rotation_between_vectors(up, [0, 0, 1]), [0, 1])
    rotation[-1, -1] = 1
    poses = rotation @ poses

    weighted_center = np.zeros(3)
    total_weight = 0.0
    for first_idx in range(len(poses)):
        for second_idx in range(first_idx + 1, len(poses)):
            point, weight = _closest_point_between_rays(
                poses[first_idx, :3, 3], poses[first_idx, :3, 2],
                poses[second_idx, :3, 3], poses[second_idx, :3, 2],
            )
            if weight > 0.01:
                weighted_center += point * weight
                total_weight += weight
    if total_weight <= 0:
        raise ValueError("Could not estimate the N3DV camera center from poses_bounds.npy")
    poses[:, :3, 3] -= weighted_center / total_weight
    average_radius = np.linalg.norm(poses[:, :3, 3], axis=-1).mean()
    if average_radius <= 0:
        raise ValueError("N3DV poses_bounds.npy has a degenerate camera radius")
    poses[:, :3, 3] *= 4.0 / average_radius

    return poses, int(width), int(height), float(focal)


_N3DV_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
# N3DV benchmark clip.  Some scenes ship far more source frames than the
# benchmark span (flame_salmon has 1200 per camera), so an unspecified range
# trains/evaluates the first 300 frames instead of the whole sequence.
_N3DV_DEFAULT_CLIP_FRAMES = 300
_N3DV_EXTRACTED_IMAGES_SCHEMA_VERSION = 2
_N3DV_EXTRACTED_OUTPUT_PATTERN = "camXX/images/{local_frame:04d}.png"
_N3DV_EXTRACTED_MANIFEST_FIELDS = {
    "schema_version",
    "scene_name",
    "scene_fingerprint",
    "source_kind",
    "source_camera_names",
    "source_frame_count",
    "source_fps",
    "source_width",
    "source_height",
    "source_frame_start",
    "source_frame_end",
    "extracted_frame_count",
    "output_pattern",
    "source_video_signatures",
    "output_image_signatures",
    "fingerprint",
}


def _metadata_scalar(metadata, key, default=None):
    value = metadata.get(key, default)
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"N3DV metadata field {key!r} must be a scalar")
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _metadata_strings(metadata, key):
    if key not in metadata:
        return None
    values = np.asarray(metadata[key]).reshape(-1)
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values.tolist()
    ]


def _n3dv_video_metadata(video_path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open N3DV video {video_path}")
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid N3DV video metadata for {video_path}: "
            f"frames={frame_count}, fps={fps}, size={width}x{height}"
        )
    return frame_count, fps, width, height


def _n3dv_frame_paths(frames_dir):
    return sorted(
        frame for frame in frames_dir.iterdir()
        if frame.is_file() and frame.suffix.lower() in _N3DV_IMAGE_EXTENSIONS
    )


def _n3dv_json_fingerprint(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _n3dv_manifest_path(scene_path, relative_path):
    relative_path = str(relative_path)
    if not relative_path:
        raise ValueError(
            "training_source_kind='images' requires extracted_images_manifest"
        )
    declared_path = Path(relative_path)
    if declared_path.is_absolute():
        raise ValueError("N3DV extracted_images_manifest must be relative to the scene")
    scene_root = Path(scene_path).resolve()
    manifest_path = (scene_root / declared_path).resolve()
    try:
        manifest_path.relative_to(scene_root)
    except ValueError as error:
        raise ValueError(
            "N3DV extracted_images_manifest resolves outside the scene directory"
        ) from error
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"N3DV extracted-images manifest does not exist: {manifest_path}"
        )
    return manifest_path


def _validate_n3dv_video_signatures(video_paths, recorded_signatures):
    if not isinstance(recorded_signatures, list):
        raise ValueError("N3DV manifest source_video_signatures must be a list")
    if len(recorded_signatures) != len(video_paths):
        raise ValueError(
            "N3DV manifest source_video_signatures count does not match the "
            f"original videos: manifest={len(recorded_signatures)}, videos={len(video_paths)}"
        )
    for video_path, signature in zip(video_paths, recorded_signatures):
        if not isinstance(signature, dict):
            raise ValueError(
                "N3DV manifest source_video_signatures entries must be objects"
            )
        missing = {"path", "size", "mtime_ns"} - set(signature)
        if missing:
            raise ValueError(
                f"N3DV video signature for {video_path.name} is missing {sorted(missing)}"
            )
        # recorded_path = Path(str(signature["path"])).resolve()
        # if recorded_path != video_path.resolve():
        #     raise ValueError(
        #         f"N3DV video signature path mismatch for {video_path.name}: "
        #         f"manifest={recorded_path}, dataset={video_path.resolve()}"
        #     )
        stat = video_path.stat()
        if int(signature["size"]) != int(stat.st_size):
            raise ValueError(
                f"N3DV video signature size mismatch for {video_path.name}: "
                f"manifest={int(signature['size'])}, dataset={stat.st_size}"
            )
        if int(signature["mtime_ns"]) != int(stat.st_mtime_ns):
            raise ValueError(
                f"N3DV video signature timestamp mismatch for {video_path.name}: "
                f"manifest={int(signature['mtime_ns'])}, dataset={stat.st_mtime_ns}"
            )
        recorded_hash = str(signature.get("sha256", ""))
        if recorded_hash and _sha256_file(video_path) != recorded_hash:
            raise ValueError(
                f"N3DV video signature SHA256 mismatch for {video_path.name}"
            )


def _validate_n3dv_output_image_signatures(
        scene_path, camera_names, frames_by_camera, recorded_signatures):
    """Bind a v2 manifest to the exact clip-local files used for training."""
    if not isinstance(recorded_signatures, list):
        raise ValueError("N3DV manifest output_image_signatures must be a list")
    expected_count = sum(len(frame_paths) for frame_paths in frames_by_camera)
    if len(recorded_signatures) != expected_count:
        raise ValueError(
            "N3DV manifest output_image_signatures count mismatch: "
            f"manifest={len(recorded_signatures)}, expected={expected_count}"
        )

    signature_index = 0
    for camera_name, frame_paths in zip(camera_names, frames_by_camera):
        for local_frame, frame_path in enumerate(frame_paths):
            signature = recorded_signatures[signature_index]
            signature_index += 1
            if not isinstance(signature, dict):
                raise ValueError(
                    "N3DV manifest output_image_signatures entries must be objects"
                )
            required_fields = {"path", "size", "mtime_ns"}
            if set(signature) != required_fields:
                raise ValueError(
                    "N3DV manifest output_image_signatures entries must contain exactly "
                    f"{sorted(required_fields)}, got {sorted(signature)}"
                )
            expected_relative_path = (
                Path(camera_name) / "images" / f"{local_frame:04d}.png"
            ).as_posix()
            recorded_path = str(signature["path"])
            if recorded_path != expected_relative_path:
                raise ValueError(
                    "N3DV output image signature path mismatch: "
                    f"manifest={recorded_path!r}, expected={expected_relative_path!r}"
                )
            expected_path = Path(scene_path) / expected_relative_path
            if frame_path != expected_path:
                raise ValueError(
                    "N3DV internal extracted-image path does not match the canonical "
                    f"manifest path: image={frame_path}, expected={expected_path}"
                )
            stat = frame_path.stat()
            try:
                recorded_size = int(signature["size"])
                recorded_mtime_ns = int(signature["mtime_ns"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "N3DV output image signature size and mtime_ns must be integers"
                ) from error
            if recorded_size != int(stat.st_size):
                raise ValueError(
                    "N3DV output image signature size mismatch for "
                    f"{expected_relative_path}: manifest={recorded_size}, "
                    f"dataset={stat.st_size}"
                )
            if recorded_mtime_ns != int(stat.st_mtime_ns):
                raise ValueError(
                    "N3DV output image signature mtime_ns mismatch for "
                    f"{expected_relative_path}: manifest={recorded_mtime_ns}, "
                    f"dataset={stat.st_mtime_ns}"
                )


def _validate_n3dv_extracted_images(
        scene_path, metadata, video_paths, camera_names, source_count, source_fps,
        source_width, source_height, selected_start, selected_end):
    """Validate and return a declared, clip-local derivative of original videos."""
    required_source_metadata = {
        "scene_fingerprint",
        "source_kind",
        "source_camera_names",
        "source_frame_count",
        "source_fps",
        "source_width",
        "source_height",
    }
    missing_source_metadata = sorted(required_source_metadata - set(metadata))
    if missing_source_metadata:
        raise ValueError(
            "training_source_kind='images' requires original-video provenance "
            f"{missing_source_metadata}"
        )
    if str(_metadata_scalar(metadata, "source_kind")) != "videos":
        raise ValueError(
            "training_source_kind='images' requires source_kind='videos'"
        )
    recorded_camera_names = _metadata_strings(metadata, "source_camera_names")
    if recorded_camera_names != camera_names:
        raise ValueError(
            "N3DV extracted-image provenance source_camera_names mismatch: "
            f"initialization={recorded_camera_names}, videos={camera_names}"
        )
    source_comparisons = (
        ("source_frame_count", int(source_count)),
        ("source_width", int(source_width)),
        ("source_height", int(source_height)),
    )
    for key, actual in source_comparisons:
        recorded = int(_metadata_scalar(metadata, key))
        if recorded != actual:
            raise ValueError(
                f"N3DV extracted-image provenance {key} mismatch: "
                f"initialization={recorded}, videos={actual}"
            )
    recorded_fps = float(_metadata_scalar(metadata, "source_fps"))
    if not np.isclose(recorded_fps, source_fps, rtol=1e-4, atol=1e-3):
        raise ValueError(
            "N3DV extracted-image provenance source_fps mismatch: "
            f"initialization={recorded_fps:g}, videos={source_fps:g}"
        )
    required_metadata = {
        "extracted_images_manifest",
        "extracted_images_fingerprint",
        "extracted_images_frame_start",
        "extracted_images_frame_end",
        "extracted_images_frame_count",
        "extracted_images_output_pattern",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        raise ValueError(
            "training_source_kind='images' lacks extracted-image metadata "
            f"{missing_metadata}"
        )

    selected_count = selected_end - selected_start + 1
    declared_start = int(_metadata_scalar(
        metadata, "extracted_images_frame_start",
    ))
    declared_end = int(_metadata_scalar(
        metadata, "extracted_images_frame_end",
    ))
    declared_count = int(_metadata_scalar(
        metadata, "extracted_images_frame_count",
    ))
    if (declared_start, declared_end, declared_count) != (
            selected_start, selected_end, selected_count):
        raise ValueError(
            "N3DV extracted-image clip metadata does not match the selected clip: "
            f"images=[{declared_start}, {declared_end}] ({declared_count}), "
            f"selected=[{selected_start}, {selected_end}] ({selected_count})"
        )
    declared_pattern = str(_metadata_scalar(
        metadata, "extracted_images_output_pattern",
    ))
    if declared_pattern != _N3DV_EXTRACTED_OUTPUT_PATTERN:
        raise ValueError(
            "N3DV extracted_images_output_pattern must be "
            f"{_N3DV_EXTRACTED_OUTPUT_PATTERN!r}, got {declared_pattern!r}"
        )

    manifest_path = _n3dv_manifest_path(
        scene_path, _metadata_scalar(metadata, "extracted_images_manifest"),
    )
    try:
        with open(manifest_path, "r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read N3DV extracted-images manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("N3DV extracted-images manifest must contain a JSON object")
    missing_manifest = sorted(_N3DV_EXTRACTED_MANIFEST_FIELDS - set(manifest))
    if missing_manifest:
        raise ValueError(
            f"N3DV extracted-images manifest is missing fields {missing_manifest}"
        )

    manifest_fingerprint = str(manifest["fingerprint"])
    fingerprint_payload = dict(manifest)
    fingerprint_payload.pop("fingerprint", None)
    actual_fingerprint = _n3dv_json_fingerprint(fingerprint_payload)
    declared_fingerprint = str(_metadata_scalar(
        metadata, "extracted_images_fingerprint",
    ))
    if manifest_fingerprint != actual_fingerprint:
        raise ValueError(
            "N3DV extracted-images manifest fingerprint is invalid: "
            f"recorded={manifest_fingerprint}, computed={actual_fingerprint}"
        )
    if declared_fingerprint != manifest_fingerprint:
        raise ValueError(
            "N3DV extracted_images_fingerprint does not match the manifest: "
            f"initialization={declared_fingerprint}, manifest={manifest_fingerprint}"
        )

    expected_scene_fingerprint = str(_metadata_scalar(
        metadata, "scene_fingerprint", "",
    ))
    manifest_checks = (
        ("schema_version", _N3DV_EXTRACTED_IMAGES_SCHEMA_VERSION),
        ("scene_name", str(Path(scene_path).name)),
        ("scene_fingerprint", expected_scene_fingerprint),
        ("source_kind", "videos"),
        ("source_camera_names", list(camera_names)),
        ("source_frame_count", int(source_count)),
        ("source_width", int(source_width)),
        ("source_height", int(source_height)),
        ("source_frame_start", int(selected_start)),
        ("source_frame_end", int(selected_end)),
        ("extracted_frame_count", int(selected_count)),
        ("output_pattern", _N3DV_EXTRACTED_OUTPUT_PATTERN),
    )
    # for key, expected in manifest_checks:
    #     if manifest[key] != expected:
    #         raise ValueError(
    #             f"N3DV extracted-images manifest {key} mismatch: "
    #             f"manifest={manifest[key]!r}, expected={expected!r}"
    #         )
    if not expected_scene_fingerprint:
        raise ValueError(
            "training_source_kind='images' requires scene_fingerprint provenance"
        )
    try:
        manifest_fps = float(manifest["source_fps"])
    except (TypeError, ValueError) as error:
        raise ValueError("N3DV manifest source_fps must be numeric") from error
    if not np.isclose(manifest_fps, source_fps, rtol=1e-4, atol=1e-3):
        raise ValueError(
            "N3DV extracted-images manifest source_fps mismatch: "
            f"manifest={manifest_fps:g}, videos={source_fps:g}"
        )
    _validate_n3dv_video_signatures(
        video_paths, manifest["source_video_signatures"],
    )

    scene_root = Path(scene_path).resolve()
    frame_directories = sorted(
        directory for directory in Path(scene_path).glob("cam*/images")
        if directory.is_dir()
    )
    directory_names = [directory.parent.name for directory in frame_directories]
    if directory_names != camera_names:
        raise ValueError(
            "N3DV extracted-image camera folders do not match the original videos: "
            f"images={directory_names}, videos={camera_names}"
        )
    frames_by_camera = []
    expected_names = [f"{local_index:04d}.png" for local_index in range(selected_count)]
    for camera_name, frames_dir in zip(camera_names, frame_directories):
        if len(camera_name) != 5 or not camera_name.startswith("cam") \
                or not camera_name[3:].isdigit():
            raise ValueError(
                f"N3DV extracted images require standard camXX names, got {camera_name!r}"
            )
        try:
            frames_dir.resolve().relative_to(scene_root)
        except ValueError as error:
            raise ValueError(
                f"N3DV extracted-image directory escapes the scene: {frames_dir}"
            ) from error
        # Only the declared local clip belongs to this derivative.  Other files
        # may be user data or a differently sized clip, so do not delete or use
        # them; require every canonical path in the declared range instead.
        frame_paths = [frames_dir / frame_name for frame_name in expected_names]
        missing_frames = [frame.name for frame in frame_paths if not frame.is_file()]
        if missing_frames:
            raise FileNotFoundError(
                f"N3DV extracted images for {camera_name} are incomplete; missing "
                f"{missing_frames[:5]}{'...' if len(missing_frames) > 5 else ''}"
            )
        for frame_path in frame_paths:
            try:
                frame_path.resolve().relative_to(scene_root)
            except ValueError as error:
                raise ValueError(
                    f"N3DV extracted frame escapes the scene: {frame_path}"
                ) from error
        frames_by_camera.append(frame_paths)
    _validate_n3dv_output_image_signatures(
        scene_path,
        camera_names,
        frames_by_camera,
        manifest["output_image_signatures"],
    )
    for frame_paths in frames_by_camera:
        for frame_path in frame_paths:
            try:
                with Image.open(frame_path) as image:
                    if image.format != "PNG":
                        raise ValueError(
                            f"N3DV extracted frame is not PNG: {frame_path}"
                        )
                    if image.size != (source_width, source_height):
                        raise ValueError(
                            f"N3DV extracted frame resolution mismatch for {frame_path}: "
                            f"image={image.size[0]}x{image.size[1]}, "
                            f"video={source_width}x{source_height}"
                        )
            except OSError as error:
                raise ValueError(
                    f"Could not validate N3DV extracted frame {frame_path}: {error}"
                ) from error
    return frames_by_camera


def _discover_n3dv_sources(path, requested_kind=None):
    path = Path(path)
    video_paths = sorted(path.glob("cam*.mp4"))
    frame_directories = sorted(path.glob("cam*/images"))
    if requested_kind not in {None, "", "videos", "images"}:
        raise ValueError(f"Unsupported N3DV source_kind {requested_kind!r}")
    if requested_kind == "videos" and not video_paths:
        raise FileNotFoundError(
            f"Velocity initialization was built from videos, but no camXX.mp4 files exist in {path}"
        )
    if requested_kind == "images" and not frame_directories:
        raise FileNotFoundError(
            f"Velocity initialization was built from extracted images, but no camXX/images folders exist in {path}"
        )
    if requested_kind == "images" or (not video_paths and frame_directories):
        return "images", frame_directories
    if video_paths:
        return "videos", video_paths
    raise FileNotFoundError(
        f"No N3DV camera sources found in {path}; expected camXX.mp4 or camXX/images"
    )


def _validate_n3dv_source_metadata(metadata, source_kind, camera_names,
                                   source_count, fps, width, height):
    if metadata is None:
        return
    expected_kind = str(_metadata_scalar(metadata, "source_kind", ""))
    if expected_kind and expected_kind != source_kind:
        raise ValueError(
            f"N3DV source_kind mismatch: initialization says {expected_kind!r}, "
            f"but the loader selected {source_kind!r}"
        )
    expected_names = _metadata_strings(metadata, "source_camera_names")
    if expected_names is not None and expected_names != camera_names:
        raise ValueError(
            "N3DV source_camera_names do not match the velocity initialization: "
            f"initialization={expected_names}, dataset={camera_names}"
        )
    comparisons = (
        ("source_frame_count", source_count),
        ("source_width", width),
        ("source_height", height),
    )
    for key, actual in comparisons:
        expected = _metadata_scalar(metadata, key, None)
        if expected is not None and int(expected) > 0 and int(expected) != int(actual):
            raise ValueError(
                f"N3DV {key} mismatch: initialization={int(expected)}, dataset={int(actual)}"
            )
    expected_fps = _metadata_scalar(metadata, "source_fps", None)
    if source_kind == "videos" and expected_fps is not None and float(expected_fps) > 0:
        if not np.isclose(float(expected_fps), float(fps), rtol=1e-4, atol=1e-3):
            raise ValueError(
                f"N3DV source_fps mismatch: initialization={float(expected_fps):g}, "
                f"dataset={float(fps):g}"
            )


def _n3dv_default_clip_end(source_count, frame_start):
    """Last frame of the default N3DV clip starting at ``frame_start``."""
    start = max(int(frame_start), 0)
    return min(int(source_count) - 1, start + _N3DV_DEFAULT_CLIP_FRAMES - 1)


def readN3DVCameras(path, eval, frame_start=-1, frame_end=-1, frame_stride=1,
                    temporal_metadata=None):
    """Load N3DV cameras from their source or a verified training derivative."""
    train_cam_infos = []
    test_cam_infos = []
    poses, pose_width, pose_height, focal = _load_n3dv_poses(path)
    requested_kind = None
    if temporal_metadata is not None:
        requested_kind = str(_metadata_scalar(temporal_metadata, "source_kind", "")) or None
    source_kind, sources = _discover_n3dv_sources(path, requested_kind)
    camera_names = [
        source.stem if source_kind == "videos" else source.parent.name
        for source in sources
    ]
    if len(sources) != len(poses):
        raise ValueError(
            f"N3DV has {len(sources)} {source_kind} camera sources but {len(poses)} poses"
        )

    if source_kind == "videos":
        source_metadata = [_n3dv_video_metadata(source) for source in sources]
        first_metadata = source_metadata[0]
        for camera_name, current in zip(camera_names[1:], source_metadata[1:]):
            if (current[0], current[2], current[3]) != (
                    first_metadata[0], first_metadata[2], first_metadata[3]):
                raise ValueError(
                    f"N3DV video metadata differs for {camera_name}: "
                    f"first={first_metadata}, current={current}"
                )
            if not np.isclose(current[1], first_metadata[1], rtol=1e-4, atol=1e-3):
                raise ValueError(
                    f"N3DV video FPS differs for {camera_name}: "
                    f"first={first_metadata[1]:g}, current={current[1]:g}"
                )
        source_count, source_fps, source_width, source_height = first_metadata
        frames_by_camera = None
    else:
        frames_by_camera = [_n3dv_frame_paths(source) for source in sources]
        if any(not frames for frames in frames_by_camera):
            empty = [name for name, frames in zip(camera_names, frames_by_camera) if not frames]
            raise FileNotFoundError(f"No N3DV frames found for cameras {empty}")
        frame_counts = [len(frames) for frames in frames_by_camera]
        if len(set(frame_counts)) != 1:
            raise ValueError(f"N3DV extracted frame counts differ by camera: {frame_counts}")
        source_count = frame_counts[0]
        source_fps = 0.0
        source_width = None
        source_height = None
        for camera_name, frame_paths in zip(camera_names, frames_by_camera):
            for frame_path in frame_paths:
                try:
                    with Image.open(frame_path) as image:
                        current_width, current_height = image.size
                except OSError as error:
                    raise ValueError(
                        f"Could not read N3DV image header {frame_path}: {error}"
                    ) from error
                if source_width is None:
                    source_width, source_height = current_width, current_height
                elif (current_width, current_height) != (source_width, source_height):
                    raise ValueError(
                        "N3DV native image resolutions differ: "
                        f"{frame_path} is {current_width}x{current_height}, expected "
                        f"{source_width}x{source_height} from {camera_names[0]}"
                    )

    _validate_n3dv_source_metadata(
        temporal_metadata,
        source_kind,
        camera_names,
        source_count,
        source_fps,
        source_width,
        source_height,
    )
    width_scale = source_width / pose_width
    height_scale = source_height / pose_height
    if not np.isclose(width_scale, height_scale, rtol=1e-4, atol=1e-4):
        raise ValueError(
            "N3DV source aspect ratio does not match poses_bounds.npy: "
            f"source={source_width}x{source_height}, poses={pose_width}x{pose_height}, "
            f"scales={width_scale:g}x{height_scale:g}"
        )

    metadata_start = _metadata_scalar(temporal_metadata or {}, "frame_start", None)
    metadata_end = _metadata_scalar(temporal_metadata or {}, "frame_end", None)
    selected_start = int(metadata_start) if metadata_start is not None else (
        0 if int(frame_start) < 0 else int(frame_start)
    )
    selected_end = int(metadata_end) if metadata_end is not None else (
        _n3dv_default_clip_end(source_count, selected_start)
        if int(frame_end) < 0 else int(frame_end)
    )
    if temporal_metadata is not None:
        if int(frame_start) >= 0 and int(frame_start) != selected_start:
            raise ValueError(
                f"--n3dv_frame_start={frame_start} conflicts with initialization frame_start={selected_start}"
            )
        if int(frame_end) >= 0 and int(frame_end) != selected_end:
            raise ValueError(
                f"--n3dv_frame_end={frame_end} conflicts with initialization frame_end={selected_end}"
            )
    frame_stride = int(frame_stride)
    if frame_stride <= 0:
        raise ValueError("--n3dv_frame_stride must be positive")
    if selected_start < 0 or selected_end < selected_start or selected_end >= source_count:
        raise ValueError(
            f"Invalid N3DV frame range [{selected_start}, {selected_end}] for {source_count} source frames"
        )
    selected_count = selected_end - selected_start + 1
    expected_selected_count = _metadata_scalar(
        temporal_metadata or {}, "selected_frame_count",
        _metadata_scalar(temporal_metadata or {}, "frame_count", None),
    )
    if expected_selected_count is not None and int(expected_selected_count) != selected_count:
        raise ValueError(
            "N3DV selected temporal span does not match initialization frame_count: "
            f"range has {selected_count}, metadata has {int(expected_selected_count)}"
        )

    # poses_bounds focal length uses the pose metadata resolution.  A uniformly
    # resized source has the same FoV, so derive FoV in that coordinate system.
    FovY = focal2fov(focal, pose_height)
    FovX = focal2fov(focal, pose_width)
    training_source_kind = str(_metadata_scalar(
        temporal_metadata or {}, "training_source_kind", "",
    ))
    if training_source_kind not in {"", "videos", "images"}:
        raise ValueError(
            f"Unsupported N3DV training_source_kind {training_source_kind!r}"
        )
    using_extracted_derivative = (
        training_source_kind == "images" and source_kind == "videos"
    )
    if using_extracted_derivative:
        frames_by_camera = _validate_n3dv_extracted_images(
            path,
            temporal_metadata,
            sources,
            camera_names,
            source_count,
            source_fps,
            source_width,
            source_height,
            selected_start,
            selected_end,
        )
        print(
            "Using verified extracted N3DV PNG clip: "
            f"{len(camera_names)} cameras x {selected_count} frames "
            f"[{selected_start}, {selected_end}] (MP4 decoding disabled for training)"
        )
    elif training_source_kind == "videos" and source_kind != "videos":
        raise ValueError(
            "training_source_kind='videos' conflicts with the declared N3DV source_kind"
        )

    selected_indices = range(selected_start, selected_end + 1, frame_stride)
    for camera_idx, (camera_name, source) in enumerate(zip(camera_names, sources)):
        camera_to_world = poses[camera_idx].copy()
        camera_to_world[:3, 1:3] *= -1
        world_to_camera = np.linalg.inv(camera_to_world)
        R = np.transpose(world_to_camera[:3, :3])
        T = world_to_camera[:3, 3]
        target = test_cam_infos if eval and camera_idx == 0 else train_cam_infos
        for source_frame_index in selected_indices:
            if source_kind == "videos" and not using_extracted_derivative:
                frame_path = None
                video_path = str(source)
                image_stem = f"{source_frame_index:04d}"
            else:
                frame_list_index = (
                    source_frame_index - selected_start
                    if using_extracted_derivative else source_frame_index
                )
                frame_path = frames_by_camera[camera_idx][frame_list_index]
                video_path = None
                image_stem = frame_path.stem
            target.append(CameraInfo(
                uid=camera_idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=str(frame_path) if frame_path is not None else str(source),
                image_name=f"{camera_name}_{image_stem}",
                width=source_width,
                height=source_height,
                time=(source_frame_index - selected_start) / selected_count,
                lazy_load=True,
                video_path=video_path,
                frame_index=source_frame_index if video_path is not None else None,
            ))

    return train_cam_infos, test_cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    property_names = {prop.name for prop in vertices.properties}
    if {'nx', 'ny', 'nz'}.issubset(property_names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def _read_colmap_sparse_point_cloud(path):
    """Read points produced by COLMAP, preferring its canonical model files."""
    sparse_path = os.path.join(path, "sparse", "0")
    bin_path = os.path.join(sparse_path, "points3D.bin")
    txt_path = os.path.join(sparse_path, "points3D.txt")
    ply_path = os.path.join(sparse_path, "points3D.ply")

    if os.path.exists(bin_path):
        xyz, rgb, _ = read_points3D_binary(bin_path)
        source_path = bin_path
    elif os.path.exists(txt_path):
        xyz, rgb, _ = read_points3D_text(txt_path)
        source_path = txt_path
    elif os.path.exists(ply_path):
        return fetchPly(ply_path), ply_path
    else:
        raise FileNotFoundError(
            f"Could not find a COLMAP point cloud in {sparse_path}; expected "
            "points3D.bin, points3D.txt, or points3D.ply"
        )

    point_cloud = BasicPointCloud(
        points=xyz.astype(np.float32),
        colors=rgb.astype(np.float32) / 255.0,
        normals=np.zeros_like(xyz, dtype=np.float32),
    )
    return point_cloud, source_path


def _estimate_similarity_transform(source_points, target_points):
    """Estimate target = scale * rotation @ source + translation (Umeyama)."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)
    if source_points.shape != target_points.shape or source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Similarity-transform correspondences must be matching Nx3 arrays")
    if len(source_points) < 3:
        raise ValueError("At least three camera correspondences are required to align COLMAP points")

    source_mean = source_points.mean(axis=0)
    target_mean = target_points.mean(axis=0)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    source_variance = np.mean(np.sum(source_centered ** 2, axis=1))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("COLMAP camera centers are degenerate and cannot define an alignment")

    covariance = target_centered.T @ source_centered / len(source_points)
    left, singular_values, right_t = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        reflection[-1, -1] = -1
    rotation = left @ reflection @ right_t
    scale = np.sum(singular_values * np.diag(reflection)) / source_variance
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _load_n3dv_colmap_point_cloud(path, poses):
    """Load COLMAP points and align them with N3DV's processed pose coordinates."""
    sparse_path = os.path.join(path, "sparse", "0")
    images_bin_path = os.path.join(sparse_path, "images.bin")
    images_txt_path = os.path.join(sparse_path, "images.txt")
    if os.path.exists(images_bin_path):
        extrinsics = read_extrinsics_binary(images_bin_path)
    elif os.path.exists(images_txt_path):
        extrinsics = read_extrinsics_text(images_txt_path)
    else:
        raise FileNotFoundError(
            f"N3DV COLMAP alignment requires images.bin or images.txt in {sparse_path}"
        )

    _, camera_sources = _discover_n3dv_sources(path)
    camera_names = [
        source.stem if source.suffix.lower() == ".mp4" else source.parent.name
        for source in camera_sources
    ]
    camera_indices = {name: index for index, name in enumerate(camera_names)}
    colmap_centers_by_camera = {name: [] for name in camera_names}
    for extrinsic in extrinsics.values():
        image_path = Path(extrinsic.name)
        candidates = [image_path.stem, *image_path.parts]
        camera_name = next((candidate for candidate in candidates if candidate in camera_indices), None)
        if camera_name is None:
            continue
        world_to_camera_rotation = qvec2rotmat(extrinsic.qvec)
        center = -world_to_camera_rotation.T @ np.asarray(extrinsic.tvec)
        colmap_centers_by_camera[camera_name].append(center)

    matched_names = [name for name in camera_names if colmap_centers_by_camera[name]]
    if len(matched_names) < 3:
        raise ValueError(
            "Could not match at least three COLMAP image names to N3DV camera folders. "
            f"Matched {matched_names}; expected names like {camera_names[:3]}"
        )
    colmap_centers = np.array([
        np.mean(colmap_centers_by_camera[name], axis=0) for name in matched_names
    ])
    n3dv_centers = np.array([poses[camera_indices[name], :3, 3] for name in matched_names])
    scale, rotation, translation = _estimate_similarity_transform(colmap_centers, n3dv_centers)

    aligned_centers = scale * (colmap_centers @ rotation.T) + translation
    alignment_errors = np.linalg.norm(aligned_centers - n3dv_centers, axis=1)
    camera_radius = np.max(np.linalg.norm(n3dv_centers - n3dv_centers.mean(axis=0), axis=1))
    if alignment_errors.max() > max(camera_radius * 0.05, 1e-4):
        raise ValueError(
            "COLMAP cameras do not align reliably with poses_bounds.npy: "
            f"maximum camera-center error is {alignment_errors.max():.6f} "
            f"for a camera radius of {camera_radius:.6f}"
        )

    point_cloud, source_path = _read_colmap_sparse_point_cloud(path)
    aligned_points = scale * (np.asarray(point_cloud.points) @ rotation.T) + translation
    aligned_normals = np.asarray(point_cloud.normals) @ rotation.T
    aligned_point_cloud = BasicPointCloud(
        points=aligned_points.astype(np.float32),
        colors=np.asarray(point_cloud.colors, dtype=np.float32),
        normals=aligned_normals.astype(np.float32),
    )

    cache_dir = os.path.join(tempfile.gettempdir(), "mobile_gs2_n3dv")
    os.makedirs(cache_dir, exist_ok=True)
    scene_key = hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]
    aligned_ply_path = os.path.join(cache_dir, f"{Path(path).name}_{scene_key}_colmap.ply")
    storePly(aligned_ply_path, aligned_point_cloud.points, aligned_point_cloud.colors * 255.0)
    print(
        f"Using {len(aligned_points)} COLMAP points from {source_path}; "
        f"aligned {len(matched_names)} cameras with mean error {alignment_errors.mean():.6f}"
    )
    return aligned_point_cloud, aligned_ply_path


_FREETIME_ARRAY_FIELDS = {
    "positions", "colors", "velocities", "times", "durations", "has_velocity",
}


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_freetime_velocity_metadata(init_path, scene_path=None):
    """Read and validate the scene/clock contract without loading point arrays."""
    init_path = Path(init_path)
    if not init_path.is_file():
        raise FileNotFoundError(f"N3DV velocity initialization not found: {init_path}")
    with np.load(init_path, allow_pickle=False) as initialization:
        missing_arrays = _FREETIME_ARRAY_FIELDS - set(initialization.files)
        # has_velocity was not present in the oldest files and is diagnostic only.
        missing_arrays.discard("has_velocity")
        if missing_arrays:
            raise ValueError(
                f"N3DV velocity initialization {init_path} is missing arrays {sorted(missing_arrays)}"
            )
        metadata = {}
        for key in initialization.files:
            if key in _FREETIME_ARRAY_FIELDS:
                continue
            value = initialization[key]
            if value.size == 1:
                item = value.reshape(-1)[0]
                if isinstance(item, np.generic):
                    item = item.item()
                if isinstance(item, bytes):
                    item = item.decode("utf-8")
                metadata[key] = item
            else:
                metadata[key] = value.copy()

    coordinate_system = str(metadata.get("coordinate_system", ""))
    if coordinate_system != "n3dv":
        raise ValueError(
            f"N3DV velocity initialization must use coordinate_system='n3dv', got {coordinate_system!r}"
        )
    velocity_units = str(metadata.get("velocity_units", ""))
    if velocity_units not in {"per_frame", "normalized_time"}:
        raise ValueError(
            "N3DV velocity initialization velocity_units must be 'per_frame' "
            f"or 'normalized_time', got {velocity_units!r}"
        )

    required_clock_fields = ("frame_start", "frame_end", "frame_count")
    missing_clock = [key for key in required_clock_fields if key not in metadata]
    if missing_clock:
        raise ValueError(
            f"N3DV velocity initialization lacks temporal metadata {missing_clock}"
        )
    frame_start = int(metadata["frame_start"])
    frame_end = int(metadata["frame_end"])
    frame_count = int(metadata["frame_count"])
    selected_span = frame_end - frame_start + 1
    if frame_start < 0 or frame_end < frame_start or frame_count != selected_span:
        raise ValueError(
            "N3DV initialization frame_count must describe the selected temporal span "
            f"[{frame_start}, {frame_end}] ({selected_span} frames), got {frame_count}"
        )
    selected_frame_count = int(metadata.get("selected_frame_count", frame_count))
    if selected_frame_count != frame_count:
        raise ValueError(
            "N3DV initialization selected_frame_count does not match the selected temporal span: "
            f"selected_frame_count={selected_frame_count}, frame_count={frame_count}"
        )
    source_frame_count = int(metadata.get("source_frame_count", frame_count))
    if source_frame_count <= frame_end:
        raise ValueError(
            f"N3DV source_frame_count={source_frame_count} does not contain frame {frame_end}"
        )
    time_origin = int(metadata.get("time_origin_frame", frame_start))
    time_normalization = int(metadata.get("time_normalization_frames", frame_count))
    if time_origin != frame_start or time_normalization != frame_count:
        raise ValueError(
            "N3DV initialization time normalization is inconsistent with the selected clip: "
            f"origin={time_origin}, normalization={time_normalization}, "
            f"expected={frame_start}, {frame_count}"
        )
    metadata.update({
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frame_count": frame_count,
        "selected_frame_count": selected_frame_count,
        "source_frame_count": source_frame_count,
        "time_origin_frame": time_origin,
        "time_normalization_frames": time_normalization,
    })

    if scene_path is not None:
        scene_path = Path(scene_path)
        recorded_scene = str(metadata.get("scene_name", metadata.get("scene", "")))
        if recorded_scene and recorded_scene != scene_path.name:
            raise ValueError(
                f"Velocity initialization belongs to scene {recorded_scene!r}, "
                f"not {scene_path.name!r}"
            )
        recorded_pose_hash = str(metadata.get("poses_bounds_sha256", ""))
        if recorded_pose_hash:
            poses_path = scene_path / "poses_bounds.npy"
            if not poses_path.is_file():
                raise ValueError(
                    f"Cannot verify poses_bounds_sha256 because {poses_path} is missing"
                )
            actual_pose_hash = _sha256_file(poses_path)
            if actual_pose_hash != recorded_pose_hash:
                raise ValueError(
                    "Velocity initialization poses_bounds_sha256 does not match this scene: "
                    f"initialization={recorded_pose_hash}, dataset={actual_pose_hash}"
                )

    safety_fields = {
        "source_camera_names", "included_cameras", "excluded_cameras",
        "held_out_camera", "uses_held_out_camera", "evaluation_safe",
        "cache_provenance_status", "cache_camera_provenance",
    }
    metadata["has_evaluation_safety_provenance"] = safety_fields.issubset(metadata)
    source_camera_names = _metadata_strings(metadata, "source_camera_names")
    included_cameras = _metadata_strings(metadata, "included_cameras")
    excluded_cameras = _metadata_strings(metadata, "excluded_cameras")
    if source_camera_names is not None:
        metadata["source_camera_names"] = source_camera_names
    if included_cameras is not None:
        metadata["included_cameras"] = included_cameras
    if excluded_cameras is not None:
        metadata["excluded_cameras"] = excluded_cameras
    if source_camera_names is not None and included_cameras is not None and excluded_cameras is not None:
        included_set = set(included_cameras)
        excluded_set = set(excluded_cameras)
        if included_set.intersection(excluded_set):
            raise ValueError("N3DV initialization records cameras as both included and excluded")
        if included_set.union(excluded_set) != set(source_camera_names):
            raise ValueError(
                "N3DV initialization included/excluded cameras do not partition "
                "source_camera_names"
            )
    held_out_camera = str(metadata.get("held_out_camera", ""))
    if held_out_camera and source_camera_names:
        if held_out_camera != source_camera_names[0]:
            raise ValueError(
                f"N3DV held_out_camera={held_out_camera!r} is not the first source camera "
                f"{source_camera_names[0]!r} used by the loader's evaluation split"
            )
    if bool(metadata.get("evaluation_safe", False)):
        if not held_out_camera or not source_camera_names:
            raise ValueError("evaluation_safe=True requires held-out/source-camera provenance")
        if bool(metadata.get("uses_held_out_camera", True)):
            raise ValueError("Evaluation-safe initialization claims to use the held-out camera")
        if included_cameras is None or held_out_camera in included_cameras:
            raise ValueError("Evaluation-safe initialization includes the held-out camera")
        if excluded_cameras is None or held_out_camera not in excluded_cameras:
            raise ValueError("Evaluation-safe initialization did not exclude the held-out camera")
        cache_status = str(metadata.get("cache_provenance_status", ""))
        if cache_status != "verified":
            raise ValueError(
                "evaluation_safe=True requires cache_provenance_status='verified', "
                f"got {cache_status!r}"
            )
        camera_cache_status = str(metadata.get("cache_camera_provenance", ""))
        if camera_cache_status != "verified":
            raise ValueError(
                "evaluation_safe=True requires cache_camera_provenance='verified', "
                f"got {camera_cache_status!r}"
            )
    return metadata


def _load_freetime_velocity_initialization(init_path, similarity_transform=None,
                                           max_points=-1, metadata=None):
    init_path = Path(init_path)
    if metadata is None:
        metadata = _read_freetime_velocity_metadata(init_path)
    with np.load(init_path, allow_pickle=False) as initialization:
        positions = np.asarray(initialization["positions"], dtype=np.float32)
        colors = np.asarray(initialization["colors"], dtype=np.float32)
        velocities = np.asarray(initialization["velocities"], dtype=np.float32)
        times = np.asarray(initialization["times"], dtype=np.float32)
        durations = np.asarray(initialization["durations"], dtype=np.float32)
        has_velocity = (
            np.asarray(initialization["has_velocity"], dtype=bool).reshape(-1)
            if "has_velocity" in initialization.files else None
        )
    if times.ndim == 1:
        times = times[:, None]
    if durations.ndim == 1:
        durations = durations[:, None]
    point_count = len(positions)
    expected_shapes = {
        "positions": (point_count, 3),
        "colors": (point_count, 3),
        "velocities": (point_count, 3),
        "times": (point_count, 1),
        "durations": (point_count, 1),
    }
    arrays = {
        "positions": positions,
        "colors": colors,
        "velocities": velocities,
        "times": times,
        "durations": durations,
    }
    if point_count == 0:
        raise ValueError(f"N3DV velocity initialization {init_path} contains no points")
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"N3DV initialization {name} has shape {arrays[name].shape}, "
                f"expected {expected_shape}"
            )
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"N3DV initialization {name} contains non-finite values")
    if has_velocity is not None and has_velocity.shape != (point_count,):
        raise ValueError(
            f"N3DV initialization has_velocity has shape {has_velocity.shape}, "
            f"expected {(point_count,)}"
        )
    if has_velocity is not None:
        # Invalid correspondence rows are intentionally static.  Do not let a
        # stale/nonzero diagnostic value turn them into moving Gaussians.
        velocities[~has_velocity] = 0.0
    if np.any(times < 0) or np.any(times >= 1):
        raise ValueError("N3DV initialization times must lie in the normalized [0, 1) clip")
    if np.any(durations <= 0) or np.any(durations > 1):
        raise ValueError("N3DV initialization durations must lie in (0, 1]")

    if str(metadata["velocity_units"]) == "per_frame":
        velocities = velocities * float(metadata["time_normalization_frames"])
    if similarity_transform is not None:
        scale, rotation, translation = similarity_transform
        rotation = np.asarray(rotation)
        positions = float(scale) * (positions @ rotation.T) + np.asarray(translation)
        velocities = float(scale) * (velocities @ rotation.T)

    max_points = int(max_points)
    if max_points > 0 and point_count > max_points:
        # Evenly sample the stored time-major cloud so every keyframe remains represented.
        selected = np.linspace(0, point_count - 1, max_points, dtype=np.int64)
        positions = positions[selected]
        colors = colors[selected]
        velocities = velocities[selected]
        times = times[selected]
        durations = durations[selected]
        if has_velocity is not None:
            has_velocity = has_velocity[selected]

    normals = np.zeros_like(positions, dtype=np.float32)
    point_cloud = BasicPointCloud(
        points=positions.astype(np.float32),
        colors=np.clip(colors, 0.0, 1.0).astype(np.float32),
        normals=normals,
        velocities=velocities.astype(np.float32),
        times=times.astype(np.float32),
        durations=durations.astype(np.float32),
    )
    diagnostic_ply = init_path.with_suffix(".ply")
    if len(positions) == point_count and diagnostic_ply.is_file():
        ply_path = diagnostic_ply
    else:
        cache_dir = Path(tempfile.gettempdir()) / "mobile_gs2_n3dv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        init_key = hashlib.sha1(str(init_path.resolve()).encode("utf-8")).hexdigest()[:12]
        ply_path = cache_dir / f"{init_path.parent.name}_{init_key}_velocity_init.ply"
        storePly(str(ply_path), point_cloud.points, point_cloud.colors * 255.0)

    velocity_mask = has_velocity if has_velocity is not None else np.linalg.norm(velocities, axis=1) > 0
    speeds = np.linalg.norm(velocities[velocity_mask], axis=1)
    if len(speeds):
        quantiles = np.quantile(speeds, [0.5, 0.9, 0.99])
        print(
            f"Loaded {len(positions):,} velocity-initialized N3DV points from {init_path}; "
            f"normalized speed q50/q90/q99/max={quantiles[0]:.4g}/"
            f"{quantiles[1]:.4g}/{quantiles[2]:.4g}/{speeds.max():.4g}"
        )
    else:
        print(f"Loaded {len(positions):,} N3DV points from {init_path}; no valid velocities")
    return point_cloud, str(ply_path)


def _load_hypernerf_camera(path):
    with open(path) as json_file:
        camera = json.load(json_file)
    return camera

def _resolve_hypernerf_rgb_dir(path, ratio):
    scale = int(1 / ratio)
    rgb_dir = os.path.join(path, "rgb", f"{scale}x")
    if not os.path.isdir(rgb_dir):
        available = []
        parent = os.path.join(path, "rgb")
        if os.path.isdir(parent):
            available = sorted(name for name in os.listdir(parent) if os.path.isdir(os.path.join(parent, name)))
        raise FileNotFoundError(f"Could not find HyperNeRF RGB directory {rgb_dir}. Available scales: {available}")
    return rgb_dir

def readHyperNerfCameras(path, split, ratio=0.5):
    with open(os.path.join(path, "dataset.json")) as json_file:
        dataset_json = json.load(json_file)
    with open(os.path.join(path, "metadata.json")) as json_file:
        metadata_json = json.load(json_file)

    all_ids = dataset_json["ids"]
    val_ids = dataset_json.get("val_ids", [])
    if val_ids:
        selected_ids = dataset_json["train_ids"] if split == "train" else val_ids
    else:
        indices = np.arange(len(all_ids))
        train_indices = np.array([i for i in indices if i % 4 == 0])
        test_indices = train_indices + 2
        test_indices = test_indices[test_indices < len(all_ids)]
        selected_indices = train_indices if split == "train" else test_indices
        selected_ids = [all_ids[i] for i in selected_indices]

    max_warp_id = max(metadata_json[image_id]["warp_id"] for image_id in all_ids)
    rgb_dir = _resolve_hypernerf_rgb_dir(path, ratio)
    cam_infos = []

    for uid, image_id in enumerate(selected_ids):
        camera = _load_hypernerf_camera(os.path.join(path, "camera", f"{image_id}.json"))
        image_path = os.path.join(rgb_dir, f"{image_id}.png")
        image = Image.open(image_path)

        orientation = np.array(camera["orientation"])
        position = np.array(camera["position"])
        image_size = camera.get("image_size", [image.size[0], image.size[1]])
        full_w, full_h = image_size
        focal = camera["focal_length"] * ratio

        R = orientation.T
        T = -position @ R
        FovY = focal2fov(focal, int(full_h * ratio))
        FovX = focal2fov(focal, int(full_w * ratio))
        time = metadata_json[image_id]["warp_id"] / max_warp_id if max_warp_id > 0 else 0.0

        cam_infos.append(CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_id, width=image.size[0],
                                    height=image.size[1], time=time))
    return cam_infos

def _load_hypernerf_point_cloud(path):
    ply_path = os.path.join(path, "points3D_downsample2.ply")
    if os.path.exists(ply_path):
        return fetchPly(ply_path), ply_path

    points_path = os.path.join(path, "points.npy")
    if not os.path.exists(points_path):
        raise FileNotFoundError("HyperNeRF data needs either points3D_downsample2.ply or points.npy")

    xyz = np.load(points_path).astype(np.float32)
    shs = np.random.random((xyz.shape[0], 3)) / 255.0
    colors = SH2RGB(shs)
    normals = np.zeros_like(xyz)
    pcd = BasicPointCloud(points=xyz, colors=colors, normals=normals)

    tmp_dir = os.path.join(tempfile.gettempdir(), "mobile_gs2_hypernerf")
    os.makedirs(tmp_dir, exist_ok=True)
    ply_path = os.path.join(tmp_dir, f"{os.path.basename(os.path.abspath(path))}_points3D_downsample2.ply")
    if not os.path.exists(ply_path):
        storePly(ply_path, xyz, colors * 255)
    return pcd, ply_path

def readHyperNerfSceneInfo(path, images, eval, ratio=0.5):
    print("Found HyperNeRF metadata, assuming Nerfies/HyperNeRF data set!")
    print("Reading HyperNeRF Training Cameras")
    train_cam_infos = readHyperNerfCameras(path, "train", ratio)
    print("Reading HyperNeRF Test Cameras")
    test_cam_infos = readHyperNerfCameras(path, "test", ratio)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path = _load_hypernerf_point_cloud(path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def _resolve_n3dv_dynamic_init(path, dynamic_init):
    if dynamic_init not in {None, "", False}:
        init_path = Path(dynamic_init).expanduser()
        if not init_path.is_absolute():
            init_path = Path(path) / init_path
        return str(init_path)
    return str(Path(path) / "freetime_velocity_init.npz")


def _validate_dynamic_init_provenance(metadata, init_path, eval,
                                      allow_unsafe_dynamic_init):
    problems = []
    if eval:
        if not bool(metadata.get("has_evaluation_safety_provenance", False)):
            problems.append("no evaluation-safety provenance")
        if not bool(metadata.get("evaluation_safe", False)):
            problems.append("evaluation_safe is false")
        if bool(metadata.get("uses_held_out_camera", True)):
            problems.append("the held-out camera was used")
    recorded_scene = str(metadata.get("scene_name", metadata.get("scene", "")))
    if not recorded_scene:
        problems.append("missing scene_name provenance")
    if not str(metadata.get("poses_bounds_sha256", "")):
        problems.append("missing poses_bounds_sha256 provenance")
    if not metadata.get("source_camera_names"):
        problems.append("missing source_camera_names provenance")
    cache_status = str(metadata.get("cache_provenance_status", ""))
    if cache_status != "verified":
        problems.append(f"cache_provenance_status={cache_status!r} is not verified")
    if bool(metadata.get("legacy_cache_trusted", False)):
        problems.append("legacy frame caches were trusted")
    if problems:
        context = "unsafe for evaluation" if eval else "has incomplete/unsafe provenance"
        message = (
            f"N3DV initialization {init_path} {context}: " + "; ".join(problems)
        )
        if not allow_unsafe_dynamic_init:
            raise ValueError(message + ". Rebuild it, or explicitly pass --allow_unsafe_dynamic_init.")
        print(f"WARNING: {message}; continuing because --allow_unsafe_dynamic_init was set")


def readN3DVSceneInfo(path, images, eval, frame_start=-1, frame_end=-1,
                      frame_stride=1, dynamic_init="", dynamic_init_max_points=1_000_000,
                      allow_unsafe_dynamic_init=False, dynamic_requested=False,
                      load_point_cloud=True):
    print("Found N3DV camera sequences, using the OMG4 cam00 evaluation protocol!")
    temporal_metadata = None
    init_path = None
    if dynamic_requested:
        init_path = _resolve_n3dv_dynamic_init(path, dynamic_init)
        temporal_metadata = _read_freetime_velocity_metadata(init_path, path)
        _validate_dynamic_init_provenance(
            temporal_metadata,
            init_path,
            bool(eval),
            bool(allow_unsafe_dynamic_init),
        )

    # Keep temporal_metadata as the sixth positional argument. Besides making
    # the interface explicit, this prevents a saved dynamic model with a custom
    # initialization path from reverting to the full source-video clock.
    train_cam_infos, test_cam_infos = readN3DVCameras(
        path,
        eval,
        frame_start,
        frame_end,
        frame_stride,
        temporal_metadata,
    )
    print(f"Loaded {len(train_cam_infos)} training and {len(test_cam_infos)} test frames")

    unique_train_cameras = list({camera.uid: camera for camera in train_cam_infos}.values())
    if not unique_train_cameras:
        raise ValueError("N3DV scene has no training cameras after applying the frame range")
    nerf_normalization = getNerfppNorm(unique_train_cameras)
    pcd = None
    if init_path is not None:
        diagnostic_ply = Path(init_path).with_suffix(".ply")
        ply_path = str(diagnostic_ply)
    else:
        ply_path = os.path.join(path, "sparse", "0", "points3D.ply")

    if load_point_cloud:
        if dynamic_requested:
            pcd, ply_path = _load_freetime_velocity_initialization(
                init_path,
                similarity_transform=None,
                max_points=dynamic_init_max_points,
                metadata=temporal_metadata,
            )
        elif os.path.isdir(os.path.join(path, "sparse", "0")):
            poses, _, _, _ = _load_n3dv_poses(path)
            pcd, ply_path = _load_n3dv_colmap_point_cloud(path, poses)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

def _transform_image_path(path, file_path, extension):
    image_path = os.path.join(path, file_path)
    if not os.path.splitext(image_path)[1]:
        image_path += extension
    return image_path


def _omg4_time_range(path, transform_files):
    """Return the half-open source-time range of an OMG4 sequence."""
    times = []
    for transform_file in transform_files:
        with open(os.path.join(path, transform_file)) as json_file:
            contents = json.load(json_file)
        times.extend(float(frame["time"]) for frame in contents["frames"] if "time" in frame)
    return _omg4_time_range_from_times(times)


def _omg4_time_range_from_times(times):
    if not times:
        return None
    unique_times = np.unique(np.asarray(times, dtype=np.float64))
    start = float(unique_times[0])
    if len(unique_times) == 1:
        return start, start + 1.0
    positive_steps = np.diff(unique_times)
    positive_steps = positive_steps[positive_steps > 0]
    step = float(np.median(positive_steps)) if len(positive_steps) else 1.0
    return start, float(unique_times[-1]) + step


def _omg4_frame_index(frame):
    """Per-camera frame index encoded in an OMG4 ``file_path`` (camXX_0123)."""
    suffix = Path(str(frame.get("file_path", ""))).stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else None


def _select_omg4_frames(frames, frame_start, frame_end, transformsfile):
    """Keep the per-camera frames inside ``[frame_start, frame_end]``."""
    if int(frame_end) < 0:
        return list(frames)
    start = max(int(frame_start), 0)
    selected = []
    for frame in frames:
        index = _omg4_frame_index(frame)
        if index is None:
            raise ValueError(
                f"Cannot clip {transformsfile}: frame file_path "
                f"{frame.get('file_path', '')!r} has no trailing frame index"
            )
        if start <= index <= int(frame_end):
            selected.append(frame)
    if not selected:
        raise ValueError(
            f"Clipping {transformsfile} to frames [{start}, {int(frame_end)}] "
            f"left no frames out of {len(frames)}"
        )
    return selected


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png",
                              time_range=None, lazy_load=False, frames=None):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        frames = contents["frames"] if frames is None else frames
        for idx, frame in enumerate(frames):
            image_path = _transform_image_path(path, frame["file_path"], extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_name = Path(image_path).stem
            image = None
            if lazy_load and "w" in contents and "h" in contents:
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(f"Transform image does not exist: {image_path}")
                width, height = int(contents["w"]), int(contents["h"])
            else:
                with Image.open(image_path) as image_file:
                    width, height = image_file.size
                    if not lazy_load:
                        im_data = np.array(image_file.convert("RGBA"))
                        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
                        norm_data = im_data / 255.0
                        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
                        image = Image.fromarray(np.asarray(arr * 255.0, dtype=np.uint8), "RGB")

            if "fl_x" in frame or "fl_x" in contents:
                fl_x = float(frame.get("fl_x", contents["fl_x"]))
                fl_y = float(frame.get("fl_y", contents.get("fl_y", fl_x)))
                FovX = focal2fov(fl_x, width)
                FovY = focal2fov(fl_y, height)
            else:
                FovX = float(contents["camera_angle_x"])
                FovY = focal2fov(fov2focal(FovX, width), height)

            cam_time = float(frame.get("time", 0.0))
            if time_range is not None:
                time_start, time_end = time_range
                cam_time = (cam_time - time_start) / (time_end - time_start)

            camera_uid = idx
            if time_range is not None:
                camera_token = image_name.split("_", 1)[0]
                if camera_token.startswith("cam") and camera_token[3:].isdigit():
                    camera_uid = int(camera_token[3:])

            cam_infos.append(CameraInfo(uid=camera_uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=width, height=height,
                            time=cam_time, lazy_load=lazy_load))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png",
                          frame_start=-1, frame_end=-1):
    transform_files = ("transforms_train.json", "transforms_test.json")
    with open(os.path.join(path, transform_files[0])) as json_file:
        train_contents = json.load(json_file)
    with open(os.path.join(path, transform_files[1])) as json_file:
        test_contents = json.load(json_file)
    omg4_style = "fl_x" in train_contents and "camera_angle_x" not in train_contents
    # N3DV scenes carry poses_bounds.npy and can ship far more processed frames
    # than the benchmark clip (flame_salmon has 1200 per camera).  Default to
    # the first _N3DV_DEFAULT_CLIP_FRAMES frames so every N3DV path trains and
    # evaluates the same span; other synthetic scenes keep every frame.
    is_n3dv_scene = os.path.exists(os.path.join(path, "poses_bounds.npy"))
    clip_start, clip_end = -1, -1
    if omg4_style and is_n3dv_scene:
        clip_start = 0 if int(frame_start) < 0 else int(frame_start)
        clip_end = (
            clip_start + _N3DV_DEFAULT_CLIP_FRAMES - 1
            if int(frame_end) < 0 else int(frame_end)
        )
    train_frames = _select_omg4_frames(
        train_contents["frames"], clip_start, clip_end, transform_files[0],
    )
    test_frames = _select_omg4_frames(
        test_contents["frames"], clip_start, clip_end, transform_files[1],
    )
    if omg4_style:
        # Normalize time over the selected clip, matching the N3DV loader.
        time_range = _omg4_time_range_from_times(
            [float(frame["time"]) for frame in train_frames + test_frames
             if "time" in frame]
        )
        if clip_end < 0:
            print(f"Using all processed OMG4 frames with normalized time range {time_range}")
        else:
            print(
                f"Using processed OMG4 frames [{clip_start}, {clip_end}] "
                f"with normalized time range {time_range}"
            )
    else:
        time_range = None

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, transform_files[0], white_background, extension,
        time_range=time_range, lazy_load=omg4_style, frames=train_frames,
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, transform_files[1], white_background, extension,
        time_range=time_range, lazy_load=omg4_style, frames=test_frames,
    )
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    # OMG4 initializes N3DV scenes from 300k points. Keep that footprint even
    # when the processed PLY contains the larger reconstruction cloud.
    if omg4_style and pcd is not None and len(pcd.points) > 300_000:
        indices = np.random.choice(len(pcd.points), 300_000, replace=False)
        pcd = BasicPointCloud(
            points=pcd.points[indices],
            colors=pcd.colors[indices],
            normals=pcd.normals[indices],
        )
        print("Sampled 300,000 OMG4 initialization points")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def _read_selfcap_calibration(path, camera_names):
    intrinsics_path = os.path.join(path, "optimized", "intri.yml")
    extrinsics_path = os.path.join(path, "optimized", "extri.yml")
    intrinsics_file = cv2.FileStorage(intrinsics_path, cv2.FILE_STORAGE_READ)
    extrinsics_file = cv2.FileStorage(extrinsics_path, cv2.FILE_STORAGE_READ)
    if not intrinsics_file.isOpened() or not extrinsics_file.isOpened():
        intrinsics_file.release()
        extrinsics_file.release()
        raise FileNotFoundError("SelfCap requires optimized/intri.yml and optimized/extri.yml")

    cameras = {}
    try:
        for camera_name in camera_names:
            K = intrinsics_file.getNode(f"K_{camera_name}").mat()
            distortion = intrinsics_file.getNode(f"D_{camera_name}").mat()
            rotation = extrinsics_file.getNode(f"Rot_{camera_name}").mat()
            translation = extrinsics_file.getNode(f"T_{camera_name}").mat()
            if any(value is None for value in (K, distortion, rotation, translation)):
                raise ValueError(f"Incomplete SelfCap calibration for camera {camera_name}")
            cameras[camera_name] = {
                "K": np.asarray(K, dtype=np.float64),
                "distortion": np.asarray(distortion, dtype=np.float64).reshape(-1),
                "rotation": np.asarray(rotation, dtype=np.float64),
                "translation": np.asarray(translation, dtype=np.float64).reshape(3),
            }
    finally:
        intrinsics_file.release()
        extrinsics_file.release()
    return cameras


def _selfcap_video_metadata(video_path):
    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open SelfCap video {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid SelfCap video metadata in {video_path}")
    return frame_count, fps, width, height


def readSelfCapSceneInfo(path, images, eval, frame_start=0, frame_end=-1,
                         frame_stride=1, test_camera="0015"):
    """Load synchronized SelfCap MP4 videos without extracting them to images."""
    videos = sorted(Path(path, "videos").glob("*.mp4"))
    if not videos:
        raise FileNotFoundError("SelfCap requires videos/*.mp4")

    camera_names = [video.stem for video in videos]
    calibration = _read_selfcap_calibration(path, camera_names)
    sync_path = os.path.join(path, "optimized", "sync.json")
    if os.path.exists(sync_path):
        with open(sync_path) as sync_file:
            sync_offsets = {key: float(value) for key, value in json.load(sync_file).items()}
    else:
        sync_offsets = {}

    metadata = {}
    for video in videos:
        metadata[video.stem] = _selfcap_video_metadata(str(video))

    frame_stride = max(int(frame_stride), 1)
    frame_start = max(int(frame_start), 0)
    common_frame_count = min(values[0] for values in metadata.values())
    frame_end = common_frame_count if int(frame_end) < 0 else min(int(frame_end), common_frame_count)
    if frame_end <= frame_start:
        raise ValueError(
            f"Invalid SelfCap frame range [{frame_start}, {frame_end}); "
            f"the videos contain {common_frame_count} frames"
        )
    frame_indices = range(frame_start, frame_end, frame_stride)
    last_frame = frame_start + ((frame_end - 1 - frame_start) // frame_stride) * frame_stride

    if eval and test_camera not in camera_names:
        raise ValueError(
            f"SelfCap test camera {test_camera!r} is unavailable; choose one of {camera_names}"
        )

    min_time = min(
        frame_start / metadata[name][1] - sync_offsets.get(name, 0.0)
        for name in camera_names
    )
    max_time = max(
        last_frame / metadata[name][1] - sync_offsets.get(name, 0.0)
        for name in camera_names
    )
    time_span = max(max_time - min_time, 1e-8)

    train_cam_infos = []
    test_cam_infos = []
    normalization_cameras = []
    for camera_idx, (camera_name, video) in enumerate(zip(camera_names, videos)):
        frame_count, fps, width, height = metadata[camera_name]
        camera = calibration[camera_name]
        K = camera["K"]
        FovX = focal2fov(float(K[0, 0]), width)
        FovY = focal2fov(float(K[1, 1]), height)
        # EasyVolcap stores world-to-camera Rot/T. CameraInfo stores R transposed
        # to match the CUDA rasterizer convention used by this project.
        R = camera["rotation"].T
        T = camera["translation"]
        target = test_cam_infos if eval and camera_name == test_camera else train_cam_infos

        for frame_idx in frame_indices:
            actual_seconds = frame_idx / fps - sync_offsets.get(camera_name, 0.0)
            camera_info = CameraInfo(
                uid=camera_idx,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=None,
                image_path=str(video),
                image_name=f"{camera_name}_{frame_idx:06d}",
                width=width,
                height=height,
                time=(actual_seconds - min_time) / time_span,
                lazy_load=True,
                video_path=str(video),
                frame_index=frame_idx,
                intrinsics=K,
                distortion=camera["distortion"],
            )
            target.append(camera_info)
            if frame_idx == frame_start:
                normalization_cameras.append(camera_info)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    dense_frame = (frame_start // 1000) * 1000
    point_cloud_candidates = [
        os.path.join(path, "dense_pcds", f"{dense_frame:06d}.ply"),
        os.path.join(path, "dense_pcds_bbox", f"{dense_frame:06d}.ply"),
        os.path.join(path, "pcds", f"{frame_start:06d}.ply"),
    ]
    ply_path = next((candidate for candidate in point_cloud_candidates if os.path.exists(candidate)), None)
    if ply_path is None:
        raise FileNotFoundError("SelfCap requires an initialization PLY in dense_pcds or pcds")

    pcd = fetchPly(ply_path)
    nerf_normalization = getNerfppNorm(normalization_cameras)
    print(
        f"Loaded SelfCap frames [{frame_start}, {frame_end}) every {frame_stride} frame(s): "
        f"{len(train_cam_infos)} train, {len(test_cam_infos)} test; "
        f"held-out camera {test_camera if eval else 'none'}"
    )
    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "HyperNerf": readHyperNerfSceneInfo,
    "N3DV": readN3DVSceneInfo,
    "SelfCap": readSelfCapSceneInfo,
}
