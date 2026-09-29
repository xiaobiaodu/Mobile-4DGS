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
from torch import nn
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import cv2
import threading
from collections import OrderedDict
from utils.general_utils import PILtoTorch
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    _prefetch_executor = None
    _transform_cache = {}
    _undistort_map_cache = {}
    _video_capture_local = threading.local()

    @classmethod
    def configure_prefetch(cls, max_workers):
        max_workers = int(max_workers)
        if max_workers > 0 and cls._prefetch_executor is None:
            cls._prefetch_executor = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="frame-prefetch",
            )

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 time=None, image_path=None, resolution=None, source_resolution=None,
                 video_path=None, frame_index=None, intrinsics=None, distortion=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        self.image_path = image_path
        self._image_resolution = resolution
        self._source_resolution = source_resolution or resolution
        self.video_path = video_path
        self.frame_index = frame_index
        self.intrinsics = intrinsics
        self.distortion = distortion
        self._prefetch_future = None

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if image is not None:
            self._original_image = image.clamp(0.0, 1.0).to(self.data_device)
            self.image_width = self._original_image.shape[2]
            self.image_height = self._original_image.shape[1]
            if gt_alpha_mask is not None:
                self._original_image *= gt_alpha_mask.to(self.data_device)
            else:
                self._original_image *= torch.ones(
                    (1, self.image_height, self.image_width), device=self.data_device
                )
        else:
            if image_path is None or resolution is None:
                raise ValueError("Lazy cameras require image_path and resolution")
            self._original_image = None
            self.image_width, self.image_height = resolution

        # Compatibility with camera serialization helpers.
        self.width = self.image_width
        self.height = self.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        transform_key = (
            np.asarray(R).tobytes(), np.asarray(T).tobytes(),
            float(self.FoVx), float(self.FoVy),
            np.asarray(trans).tobytes(), float(scale),
        )
        cached_transforms = self._transform_cache.get(transform_key)
        if cached_transforms is None:
            world_view_transform = torch.tensor(
                getWorld2View2(R, T, trans, scale)
            ).transpose(0, 1).cuda()
            projection_matrix = getProjectionMatrix(
                znear=self.znear, zfar=self.zfar,
                fovX=self.FoVx, fovY=self.FoVy,
            ).transpose(0, 1).cuda()
            full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
            ).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            cached_transforms = (
                world_view_transform, projection_matrix,
                full_proj_transform, camera_center,
            )
            self._transform_cache[transform_key] = cached_transforms
        (
            self.world_view_transform,
            self.projection_matrix,
            self.full_proj_transform,
            self.camera_center,
        ) = cached_transforms

    @property
    def original_image(self):
        if self._original_image is not None:
            return self._original_image

        future = self._prefetch_future
        self._prefetch_future = None
        if future is not None:
            rgb = future.result()
        else:
            rgb = self._load_image_cpu(pin_memory=False)
        return rgb.to(
            self.data_device,
            non_blocking=rgb.is_pinned() and self.data_device.type == "cuda",
        )

    def _load_image_cpu(self, pin_memory):
        if self.video_path is not None:
            rgb = self._load_video_frame_cpu()
        else:
            with Image.open(self.image_path) as image:
                resized_image = PILtoTorch(image, self._image_resolution)
            rgb = resized_image[:3, ...].clamp(0.0, 1.0)
            if resized_image.shape[0] == 4:
                rgb *= resized_image[3:4, ...]
        rgb = rgb.contiguous()
        if pin_memory and torch.cuda.is_available():
            rgb = rgb.pin_memory()
        return rgb

    def _load_video_frame_cpu(self):
        captures = getattr(self._video_capture_local, "captures", None)
        if captures is None:
            captures = OrderedDict()
            self._video_capture_local.captures = captures
        capture = captures.pop(self.video_path, None)
        if capture is None or not capture.isOpened():
            capture = cv2.VideoCapture(self.video_path)
        captures[self.video_path] = capture
        while len(captures) > 4:
            _, old_capture = captures.popitem(last=False)
            old_capture.release()
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_index))
        success, frame = capture.read()
        if not success:
            capture.release()
            capture = cv2.VideoCapture(self.video_path)
            captures[self.video_path] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(self.frame_index))
            success, frame = capture.read()
        if not success:
            raise RuntimeError(
                f"Could not decode frame {self.frame_index} from {self.video_path}"
            )

        target_width, target_height = self._image_resolution
        source_width, source_height = self._source_resolution
        if (target_width, target_height) != (source_width, source_height):
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

        if self.intrinsics is not None and self.distortion is not None:
            K = np.asarray(self.intrinsics, dtype=np.float64).copy()
            K[0, :] *= target_width / source_width
            K[1, :] *= target_height / source_height
            distortion = np.asarray(self.distortion, dtype=np.float64).reshape(-1)
            map_key = (
                self.video_path, target_width, target_height,
                K.tobytes(), distortion.tobytes(),
            )
            maps = self._undistort_map_cache.get(map_key)
            if maps is None:
                maps = cv2.initUndistortRectifyMap(
                    K, distortion, None, K,
                    (target_width, target_height), cv2.CV_32FC1,
                )
                self._undistort_map_cache[map_key] = maps
            frame = cv2.remap(
                frame, maps[0], maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)

    def prefetch(self):
        if (
            self._original_image is None
            and self._prefetch_future is None
            and self._prefetch_executor is not None
        ):
            self._prefetch_future = self._prefetch_executor.submit(
                self._load_image_cpu, True
            )

    def load_cam_parm_to_device(self, device):
        self.world_view_transform = self.world_view_transform.to(device, non_blocking=True)
        self.projection_matrix = self.projection_matrix.to(device, non_blocking=True)
        self.full_proj_transform = self.full_proj_transform.to(device, non_blocking=True)
        self.camera_center = self.camera_center.to(device, non_blocking=True)
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
