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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.selfcap_start = 0
        self.selfcap_end = -1
        self.selfcap_stride = 1
        self.selfcap_test_camera = "0015"
        self.n3dv_frame_start = 0
        self.n3dv_frame_end = 299
        self.n3dv_frame_stride = 1
        self.dynamic_init = ""
        # Match the processed OMG4/N3DV initialization budget.  The previous
        # 100k cap discarded two thirds of the velocity-bearing seed cloud
        # before densification and was especially harmful around moving edges.
        self.dynamic_init_max_points = 300_000
        # Persist the loader choice in cfg_args.  Older trained models do not
        # have this field and must keep using their transforms coordinate frame.
        # Use an integer instead of a store_true flag so this can be explicitly
        # disabled with --dynamic_n3dv_velocity_loader 0 when comparing against
        # an older processed-transforms run.
        self.dynamic_n3dv_velocity_loader = 1
        self.allow_unsafe_dynamic_init = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        # --dynamic is declared by OptimizationParams during training, then
        # persisted in cfg_args.  Keep it when ModelParams reads that cfg for
        # rendering without declaring a duplicate argparse option here.
        g.dynamic = bool(getattr(args, "dynamic", False))
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.separate_sh = True
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        
        self.mv = 1

        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025 
        self.shfeature_lr = 0.005 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.001
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # fluxgs parameters
        self.loss_thresh = 0.1
        self.grad_abs_thresh = 0.0012  
        self.highfeature_lr = 0.02
        self.lowfeature_lr = 0.0025
        self.grad_thresh = 0.0002
        self.dense = 0.001
        self.mult = 0.5      # multiplier for the compact box to control the tile number of each splat
        self.split_distance = 0.45       # Improved-GS long-axis split offset ratio
        self.opacity_reduction = 0.6     # child opacity multiplier after long-axis splitting
        


        # Staged SVQ, following OMG4: static attributes are quantized first and
        # fine-tuned, then the temporal ones, so the static codebooks do not have
        # to absorb the temporal reconstruction error in the same step.
        self.svq_itr = 28_000
        self.svq_4d_itr = 29_000
        self.svq_lr = 1e-4
        self.net_itr = 3_000
        # OMG4 trains the explicit representation to convergence before switching
        # to the implicit appearance MLP. Dynamic scenes get the long explicit
        # phase; the static default is left alone.
        self.dynamic_net_itr = 3_000
        self.importance_thresh = 0.96
        self.lambda_ld = 2.0
        self.slice_scale = 1
        self.cluster_scale = 2**9
        self.slice_rot = 2
        self.cluster_rot = 2**13
        self.slice_app = 2
        self.cluster_app = 2**10
        self.slice_dynamic = 1
        self.cluster_dynamic = 2**10

        self.shsnn_lr = 1e-2
        self.rot_feature_dim = 2
        self.rot_feature_lr = 0.0025
        self.nn_iter = 3000
        self.dynamic_nn_iter = 3_000
        self.num_mc_points = 2048
        self.pruning_quantile = 0.1
        self.importance_quantile = 0.6
        self.dynamic_score_cameras = 12
        self.dynamic_score_time_bins = 8
        self.prefetch_size = 16
        self.prefetch_workers = 4

        self.random_background = False
        self.optimizer_type = "default"
        self.dynamic = False
        self.dynamic_lr = 1e-3
        self.dynamic_velocity_lr = 1e-2
        self.dynamic_velocity_lr_final = 1e-4
        self.dynamic_velocity_max = 0.0
        self.dynamic_acceleration_lr = 2e-3
        self.dynamic_acceleration_max = 10.0
        self.dynamic_time_lr = 1e-3
        self.dynamic_duration_lr = 5e-3
        # Keep temporal optimization unchanged through 25k, smoothly decay it,
        # then freeze acceleration/time/duration at 30k to prevent late drift.
        self.dynamic_temporal_lr_decay_start = 25_000
        self.dynamic_temporal_lr_freeze_iter = 30_000
        self.dynamic_time_init = 0.5
        self.dynamic_random_time_init = 1
        self.dynamic_duration_init = 0.2
        self.dynamic_duration_min = 0.01
        # Very broad temporal Gaussians create duplicated moving edges.  Keep
        # enough support for smooth trajectories while excluding near-global
        # windows, and softly discourage widths above 0.25 below.
        self.dynamic_duration_max = 0.5
        self.dynamic_duration_soft_max = 0.25
        self.lambda_duration_reg = 5e-3
        # Bound the one-sigma spatial trail, rather than shrinking every
        # temporal window indiscriminately.  The velocity initialization has a
        # typical |v|*duration around 0.07 and a 90th percentile near 0.14.
        self.dynamic_motion_extent_soft_max = 0.15
        self.lambda_motion_extent_reg = 2e-3
        self.lambda_acceleration_reg = 1e-5
        self.lambda_edge = 2e-2
        # ExactSplit4D stage 1: a hard-concrete gate has an exactly binary
        # forward pass while retaining a straight-through gradient during the
        # decomposition phase.  The partition is committed before 4D SVQ.
        self.dynamic_gate_lr = 5e-3
        self.dynamic_gate_temperature_init = 2.0
        self.dynamic_gate_temperature_final = 0.1
        self.dynamic_gate_anneal_start = 1_000
        self.dynamic_gate_freeze_iter = 25_000
        self.dynamic_gate_gamma = -0.1
        self.dynamic_gate_zeta = 1.1
        self.dynamic_gate_static_probability = 0.15
        # Dynamic datasets without a velocity initializer start conservatively
        # all-dynamic and may be demoted by sparsity later; starting them all
        # static prevents the temporal parameters from ever receiving signal.
        self.dynamic_gate_unseeded_probability = 0.95
        self.dynamic_gate_motion_probability = 0.995
        # N3DV velocities use normalized-scene units over the full sequence.
        # A threshold of 5 keeps the strong, likely-object correspondences
        # (about 32% of the supplied cloud) while leaving low-speed matching
        # noise free to become static.
        self.dynamic_gate_motion_threshold = 5.0
        # Velocity seeds already provide the main exploration signal.  Limit
        # random hard-concrete flips to the early bootstrap so static texture is
        # not averaged through a randomly moving branch for half the training.
        self.dynamic_gate_stochastic_until = 5_000
        self.lambda_dynamic_sparsity = 2e-5
        self.lambda_gate_binary = 1e-4
        self.dynamic_sparsity_start = 5_000
        self.dynamic_sparsity_ramp_end = 20_000
        self.dynamic_binary_start = 22_000
        # A negative gate-logit gradient means that multi-time reconstruction
        # error is asking a currently static primitive to become dynamic.
        self.dynamic_promotion_warmup = 2_000
        self.dynamic_promotion_interval = 500
        self.dynamic_promotion_ema_decay = 0.99
        self.dynamic_promotion_quantile = 0.995
        self.dynamic_promotion_min_score = 1e-7
        self.dynamic_promotion_max_fraction = 0.01
        self.dynamic_promotion_logit = 8.0

        # DashGaussian-style multi-scale training: the training render target
        # starts at 1/multiscale_max_scale of the loaded camera resolution and
        # is raised back to full resolution along a frequency-energy schedule.
        # Densification itself is untouched; only the training render and its
        # ground-truth target change resolution.
        # Use an integer instead of a store_true flag so it can be explicitly
        # disabled with --multiscale 0 to reproduce the single-resolution run.
        self.multiscale = 1
        # the paper's "a": initial downsampling factor, and the frequency
        # energy floor E(1)/a used to derive it.  a=1 disables the schedule.
        self.multiscale_max_scale = 4.0
        # number of resolution levels placed between the lowest resolution and
        # the full-resolution target (m in the paper's supplementary material).
        self.multiscale_levels = 5
        # iteration that reaches full resolution.  -1 means "use
        # --densify_until_iter", so densification statistics are gathered at
        # (mostly) the resolutions the schedule visits.
        self.multiscale_full_res_iter = -1
        # training views sampled once for the DFT frequency-energy estimate.
        self.multiscale_num_images = 16
        # upper bound of the integer downsampling factors searched while
        # measuring the energy curve.
        self.multiscale_search_max_scale = 8
        # sampling seed, kept in cfg_args so a run can be reproduced.
        self.multiscale_seed = 0
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
