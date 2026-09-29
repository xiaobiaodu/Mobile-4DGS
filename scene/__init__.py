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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.compress_utils import load_comp_web


def _argument_or_default(args, name, default):
    value = getattr(args, name, default)
    return default if value is None else value


def _use_dynamic_n3dv_loader(args):
    source_path = args.source_path
    has_n3dv_source = os.path.exists(os.path.join(source_path, "poses_bounds.npy")) and (
        os.path.isfile(os.path.join(source_path, "cam00.mp4"))
        or os.path.isdir(os.path.join(source_path, "cam00", "images"))
    )
    return (
        bool(_argument_or_default(args, "dynamic", False))
        and bool(_argument_or_default(args, "dynamic_n3dv_velocity_loader", False))
        and has_n3dv_source
    )


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], decode=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "scene.json")) and \
           os.path.exists(os.path.join(args.source_path, "dataset.json")) and \
           os.path.exists(os.path.join(args.source_path, "metadata.json")):
            scene_info = sceneLoadTypeCallbacks["HyperNerf"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "optimized", "intri.yml")) and \
             os.path.exists(os.path.join(args.source_path, "optimized", "extri.yml")) and \
             os.path.isdir(os.path.join(args.source_path, "videos")):
            scene_info = sceneLoadTypeCallbacks["SelfCap"](
                args.source_path,
                args.images,
                args.eval,
                args.selfcap_start,
                args.selfcap_end,
                args.selfcap_stride,
                args.selfcap_test_camera,
            )
        # A dynamic N3DV run must consume freetime_velocity_init.npz.  The old
        # ordering selected transforms_*.json first and silently initialized
        # every velocity to zero, so the learned split was governed by the
        # promotion quota instead of motion evidence.
        elif _use_dynamic_n3dv_loader(args):
            print("Found dynamic N3DV data, using velocity-initialized loader!")
            scene_info = sceneLoadTypeCallbacks["N3DV"](
                args.source_path,
                args.images,
                args.eval,
                _argument_or_default(args, "n3dv_frame_start", -1),
                _argument_or_default(args, "n3dv_frame_end", -1),
                _argument_or_default(args, "n3dv_frame_stride", 1),
                _argument_or_default(args, "dynamic_init", ""),
                _argument_or_default(args, "dynamic_init_max_points", 1_000_000),
                _argument_or_default(args, "allow_unsafe_dynamic_init", False),
                _argument_or_default(args, "dynamic", False),
                not bool(self.loaded_iter),
            )
        # Static/legacy processed runs keep their original coordinate system.
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")) and \
             os.path.exists(os.path.join(args.source_path, "transforms_test.json")):
            print("Found processed transforms, using the OMG4 train/test split!")
            # N3DV sources can hold more frames than the benchmark clip; the
            # N3DV frame range keeps train/test/eval on the same 300-frame span.
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval,
                frame_start=_argument_or_default(args, "n3dv_frame_start", -1),
                frame_end=_argument_or_default(args, "n3dv_frame_end", -1),
            )
        elif os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")) and (
             os.path.isfile(os.path.join(args.source_path, "cam00.mp4")) or
             os.path.isdir(os.path.join(args.source_path, "cam00", "images"))):
            scene_info = sceneLoadTypeCallbacks["N3DV"](
                args.source_path,
                args.images,
                args.eval,
                _argument_or_default(args, "n3dv_frame_start", -1),
                _argument_or_default(args, "n3dv_frame_end", -1),
                _argument_or_default(args, "n3dv_frame_stride", 1),
                _argument_or_default(args, "dynamic_init", ""),
                _argument_or_default(args, "dynamic_init_max_points", 1_000_000),
                _argument_or_default(args, "allow_unsafe_dynamic_init", False),
                _argument_or_default(args, "dynamic", False),
                not bool(self.loaded_iter),
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        else:
            raise FileNotFoundError(
                f"Could not recognize scene type at {args.source_path!r}. "
                "Check the dataset path and expected metadata files."
            )

        if not self.loaded_iter:
            if scene_info.point_cloud is None:
                raise RuntimeError(
                    "The selected scene loader did not provide an initialization point cloud. "
                    "For dynamic N3DV training, run preprocessing_n3dv.py and pass --dynamic."
                )
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if any(
                getattr(cam, "video_path", None) is not None or getattr(cam, "lazy_load", False)
                for cam in camlist
            ):
                # Dynamic datasets contain thousands of frames per fixed camera.
                # cameras.json stores poses only, so avoid writing every duplicate.
                camlist = list({cam.uid: cam for cam in camlist}.values())
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            gaussians.active_sh_degree = 1
             
            if decode:
                print("decoding..........")
                save_dict = load_comp_web(args.model_path + "/comp.json")
                gaussians.rot_feature_dim = int(save_dict.get('rot_feature_dim', 2))
                gaussians.init_shsnn()
                gaussians.construct_net(train=False)
                gaussians.decode(save_dict)
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
