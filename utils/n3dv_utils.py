"""Lightweight N3DV camera-pose conversion utilities.

This module intentionally depends only on NumPy so dataset preprocessing does
not import the training-time ``scene`` package (and its CUDA/RAPIDS stack).
"""

from pathlib import Path

import numpy as np


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
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
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


def load_n3dv_poses(path):
    """Load, orient, center, and scale N3DV ``poses_bounds.npy`` cameras."""
    poses_bounds = np.load(Path(path) / "poses_bounds.npy")
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
