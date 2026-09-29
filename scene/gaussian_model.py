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

import math
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, identity_gate, cosine_decay_to_zero
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation


from utils.gpcc_utils import compress_gpcc, decompress_gpcc, calculate_morton_order, float16_to_uint16, uint16_to_float16
from utils.compress_utils import *


try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

try:
    import tinycudann as tcnn
except (ImportError, OSError):
    import warnings
    warnings.warn("tinycudann (tcnn) not found. Install with: pip install tinycudann or git+https://github.com/NVlabs/tiny-cuda-nn.git#subdirectory=bindings/torch")
    # Provide a fallback if tcnn is not available
    class MockTcnn:
        class NetworkWithInputEncoding:
            def __init__(self, *args, **kwargs): pass
        class Network:
            def __init__(self, *args, **kwargs): pass
    tcnn = MockTcnn()







class SHSNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 4*3, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.factor = 2

        self.main = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * self.factor),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * self.factor, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // self.factor),
            nn.ReLU(),
        )
        self.shs_output = nn.Sequential(
            nn.Linear(self.hidden_dim // self.factor, output_dim),
        )
        # self.opacity_output = nn.Sequential(
        #     nn.Linear(self.hidden_dim // self.factor, 1),
        #     nn.Sigmoid()
        # )
 
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_weights(self.shs_output[0], init_output=0)

    def init_weights(self, final_linear, init_output):

        nn.init.constant_(final_linear.weight, 0.0)
        nn.init.constant_(final_linear.bias, init_output)



    def forward(self, shs, opacity, scales, xyz, rotations ):
        shs = shs.view(shs.size(0), -1)
        shs = torch.nn.functional.normalize(shs)
        scales = torch.nn.functional.normalize(scales)

        feat = torch.concat([shs, opacity, scales, xyz, rotations], dim=1)
        feat = self.main(feat)

        shs_offset = self.shs_output(feat)
        # opacity = self.opacity_output(feat)
 

        return shs_offset.view(-1, 4, 3)



class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def modify_functions(self):
        old_opacities = self.get_opacity.clone()
        self.opacity_activation = torch.abs
        self.inverse_opacity_activation = identity_gate
        self._opacity = self.opacity_activation(old_opacities)

    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = sh_degree
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        # self.max_sh_rest = (sh_degree+1)**2 - 1
        self.max_sh_rest = 3
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.shoptimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()


        self.vq_enabled = False
        self.net_enabled = False
        self.shoffset_enabled = False
        self.dynamic_enabled = False
        self._velocity = torch.empty(0)
        self._acceleration = torch.empty(0)
        self._time = torch.empty(0)
        self._duration = torch.empty(0)
        self._dynamic_logit = torch.empty(0)
        self._forced_dynamic_mask = torch.empty(0, dtype=torch.bool)
        self._committed_dynamic_mask = torch.empty(0, dtype=torch.bool)
        self._promotion_ema = torch.empty(0)
        self.dynamic_gate_frozen = False
        self.dynamic_gate_training = False
        self.dynamic_gate_temperature = 1.0
        self.dynamic_gate_temperature_init = 2.0
        self.dynamic_gate_temperature_final = 0.1
        self.dynamic_gate_anneal_start = 1_000
        self.dynamic_gate_freeze_iter = 25_000
        self.dynamic_gate_stochastic_until = 5_000
        self.dynamic_gate_iteration = 0
        self.dynamic_gate_gamma = -0.1
        self.dynamic_gate_zeta = 1.1
        self.dynamic_duration_min = 1e-4
        self.dynamic_duration_max = None
        self.dynamic_temporal_lr_decay_start = 25_000
        self.dynamic_temporal_lr_freeze_iter = 30_000
        self.dynamic_temporal_base_lrs = {}

        self._features_static = torch.empty(0)
        self._features_view = torch.empty(0)
        self._features_rot = torch.empty(0)
        self.rot_feature_dim = 2
        self._contract_aabb = None
        self.dynamic_codes = []
        self.dynamic_indices = []
    



    def init_shsnn(self, training_args=None):
        self.vnn_input_dim =  3*4 + 3 + 3 + 4 + 1
   


        self.shs_nn = SHSNN(self.vnn_input_dim).cuda()
        if training_args is not None:
            l = [
                {'params': self.shs_nn.parameters(), 'lr': training_args.shsnn_lr,
                 "name": "shs_nn"},

            ]
            self.shs_nn_optimizer = torch.optim.Adam(l)

    def capture(self, optimizer_type):
        if optimizer_type == "default":
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            self.shoptimizer.state_dict() if self.shoptimizer is not None else None,
            self.spatial_lr_scale,
            self.dynamic_enabled,
            self._velocity,
            self._acceleration,
            self._time,
            self._duration,
            {
                "version": 2,
                "logit": self._dynamic_logit,
                "forced_dynamic_mask": self._forced_dynamic_mask,
                "committed_mask": self._committed_dynamic_mask,
                "promotion_ema": self._promotion_ema,
                "frozen": self.dynamic_gate_frozen,
                "temperature": self.dynamic_gate_temperature,
            },
        )
        else:
            return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.xyz_gradient_accum_abs,
            self.denom,
            self.optimizer.state_dict(),
            None,
            self.spatial_lr_scale,
            self.dynamic_enabled,
            self._velocity,
            self._acceleration,
            self._time,
            self._duration,
            {
                "version": 2,
                "logit": self._dynamic_logit,
                "forced_dynamic_mask": self._forced_dynamic_mask,
                "committed_mask": self._committed_dynamic_mask,
                "promotion_ema": self._promotion_ema,
                "frozen": self.dynamic_gate_frozen,
                "temperature": self.dynamic_gate_temperature,
            },
        )
    
    def restore(self, model_args, training_args):
        # Historical sparse-Adam checkpoints omitted the shoptimizer slot that
        # default-Adam checkpoints contained.  Insert the missing placeholder
        # before applying the common schema.
        if (
            self.optimizer_type != "default"
            and len(model_args) > 12
            and model_args[12] is not None
            and not isinstance(model_args[12], dict)
        ):
            model_args = (*model_args[:12], None, *model_args[12:])
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        xyz_gradient_accum_abs,
        denom,
        opt_dict,
        shopt_dict,
        self.spatial_lr_scale,
        *dynamic_args) = model_args
        gate_state = None
        if dynamic_args and isinstance(dynamic_args[-1], dict):
            gate_state = dynamic_args.pop()
        if dynamic_args:
            if len(dynamic_args) == 4:
                self.dynamic_enabled, self._velocity, self._time, self._duration = dynamic_args
                self._acceleration = torch.zeros_like(self._velocity)
            else:
                self.dynamic_enabled, self._velocity, self._acceleration, self._time, self._duration = dynamic_args
        elif getattr(training_args, 'dynamic', False):
            self.construct_dynamic_net(training_args)
        if gate_state is not None:
            self._dynamic_logit = gate_state.get("logit", torch.empty(0))
            self._forced_dynamic_mask = gate_state.get(
                "forced_dynamic_mask", torch.empty(0, dtype=torch.bool)
            )
            self._committed_dynamic_mask = gate_state.get(
                "committed_mask", torch.empty(0, dtype=torch.bool)
            ).reshape(-1).to(device=self._xyz.device, dtype=torch.bool)
            self._promotion_ema = gate_state.get("promotion_ema", torch.empty(0))
            self.dynamic_gate_frozen = bool(gate_state.get("frozen", False))
            self.dynamic_gate_temperature = float(gate_state.get("temperature", 1.0))
            if (
                self.dynamic_gate_frozen
                and self._committed_dynamic_mask.numel() != self._xyz.shape[0]
            ):
                raise ValueError(
                    "Frozen dynamic-gate checkpoint has "
                    f"{self._committed_dynamic_mask.numel()} committed rows for "
                    f"{self._xyz.shape[0]} Gaussians"
                )
            if self.dynamic_gate_frozen:
                self._forced_dynamic_mask = self._committed_dynamic_mask.clone()
        elif dynamic_args and self.dynamic_enabled:
            # Checkpoints written before strict gates represented every
            # temporal primitive as dynamic, even when its learned velocity was
            # close to zero.  Preserve that rendering meaning on resume.
            self._committed_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=self._xyz.device
            )
            self.dynamic_gate_frozen = False
            self._initialize_dynamic_gate(training_args, legacy_dynamic=True)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if self.shoptimizer is not None and shopt_dict is not None:
            self.shoptimizer.load_state_dict(shopt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def get_features_offset(self, shs, opacity, rotations=None, xyz=None):
        if rotations is None:
            if self.net_enabled:
                with torch.no_grad():
                    cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
                    rotations = torch.nn.functional.normalize(self.mlp_rotation(torch.cat([cont_feature, self._features_rot], dim=-1)).float(), dim=-1)
            else:
                rotations = self.get_rotation
        xyz = self.get_xyz if xyz is None else xyz
        offset = self.shs_nn(shs.view(-1, 4*3), opacity, self.get_scaling, xyz, rotations)
        return offset

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def _get_dynamic_attributes(self):
        if self.vq_enabled and hasattr(self, "dynamic_codes") and len(self.dynamic_codes) > 0:
            dynamic_attrs = []
            for i in range(len(self.dynamic_codes)):
                dynamic_attrs.append(self.dynamic_codes[i][self.dynamic_indices[i]])
            dynamic_attrs = torch.cat(dynamic_attrs, dim=-1).float()
            if dynamic_attrs.shape[1] >= 8:
                return dynamic_attrs[:, 0:3], dynamic_attrs[:, 3:6], dynamic_attrs[:, 6:7], dynamic_attrs[:, 7:8]
            return dynamic_attrs[:, 0:3], torch.zeros_like(dynamic_attrs[:, 0:3]), dynamic_attrs[:, 3:4], dynamic_attrs[:, 4:5]
        if self._acceleration.numel() == 0 or self._acceleration.shape[0] != self._velocity.shape[0]:
            acceleration = torch.zeros_like(self._velocity)
        else:
            acceleration = self._acceleration
        return self._velocity, acceleration, self._time, self._duration

    @staticmethod
    def _probability_to_logit(probability, eps=1e-6):
        probability = float(min(max(probability, eps), 1.0 - eps))
        return math.log(probability / (1.0 - probability))

    def _initialize_dynamic_gate(self, training_args=None, legacy_dynamic=False):
        """Create per-Gaussian gate logits without changing legacy renders.

        Velocity-initialized training uses motion to seed the partition.  A
        dynamic PLY/checkpoint that predates gates is treated as fully dynamic,
        preserving its old rendering semantics.
        """
        n_points = self._xyz.shape[0]
        if n_points == 0:
            self._dynamic_logit = torch.empty((0, 1), device=self._xyz.device)
            self._forced_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=self._xyz.device
            )
            self._promotion_ema = torch.empty((0, 1), device=self._xyz.device)
            return

        device = self._xyz.device
        motion_threshold = float(getattr(
            training_args, "dynamic_gate_motion_threshold", 5.0
        ))
        if legacy_dynamic:
            # Override any same-sized state left on a reused model.  Older
            # checkpoints had no gate, so all temporal rows were dynamic.
            self._forced_dynamic_mask = torch.ones(
                n_points, dtype=torch.bool, device=device
            )
        elif self._forced_dynamic_mask.numel() != n_points:
            if self._velocity.numel() == n_points * 3:
                forced_dynamic = torch.linalg.vector_norm(
                    self._velocity.detach().reshape(n_points, 3), dim=-1
                ) > motion_threshold
            else:
                forced_dynamic = torch.zeros(
                    n_points, dtype=torch.bool, device=device
                )
            self._forced_dynamic_mask = forced_dynamic
        else:
            self._forced_dynamic_mask = self._forced_dynamic_mask.reshape(-1).to(
                device=device, dtype=torch.bool
            )

        if legacy_dynamic:
            self._dynamic_logit = torch.full(
                (n_points, 1),
                self._probability_to_logit(0.95),
                dtype=torch.float,
                device=device,
            )
        elif self._dynamic_logit.numel() == n_points:
            self._dynamic_logit = self._dynamic_logit.reshape(n_points, 1).to(
                device=device, dtype=torch.float
            )
        else:
            static_probability = float(getattr(
                training_args, "dynamic_gate_static_probability", 0.05
            ))
            unseeded_probability = float(getattr(
                training_args, "dynamic_gate_unseeded_probability", 0.95
            ))
            motion_probability = float(getattr(
                training_args, "dynamic_gate_motion_probability", 0.95
            ))
            has_motion_seed = bool(self._forced_dynamic_mask.any())
            base_probability = (
                static_probability if has_motion_seed else unseeded_probability
            )
            static_logit = self._probability_to_logit(base_probability)
            motion_logit = self._probability_to_logit(motion_probability)
            logits = torch.full(
                (n_points, 1), static_logit, dtype=torch.float, device=device
            )
            if self._velocity.numel() == n_points * 3:
                logits[self._forced_dynamic_mask] = motion_logit
            self._dynamic_logit = logits

        if self._promotion_ema.numel() != n_points:
            self._promotion_ema = torch.zeros(
                (n_points, 1), dtype=torch.float, device=device
            )
        else:
            self._promotion_ema = self._promotion_ema.reshape(n_points, 1).to(
                device=device, dtype=torch.float
            )

    def _configure_dynamic_gate(self, training_args=None):
        self.dynamic_gate_temperature_init = float(getattr(
            training_args, "dynamic_gate_temperature_init", 2.0
        ))
        self.dynamic_gate_temperature_final = float(getattr(
            training_args, "dynamic_gate_temperature_final", 0.1
        ))
        self.dynamic_gate_anneal_start = int(getattr(
            training_args, "dynamic_gate_anneal_start", 1_000
        ))
        self.dynamic_gate_freeze_iter = int(getattr(
            training_args, "dynamic_gate_freeze_iter", 25_000
        ))
        self.dynamic_gate_stochastic_until = int(getattr(
            training_args, "dynamic_gate_stochastic_until", 5_000
        ))
        self.dynamic_gate_gamma = float(getattr(
            training_args, "dynamic_gate_gamma", -0.1
        ))
        self.dynamic_gate_zeta = float(getattr(
            training_args, "dynamic_gate_zeta", 1.1
        ))
        if not self.dynamic_gate_gamma < 0.0 < self.dynamic_gate_zeta:
            raise ValueError(
                "hard-concrete bounds require dynamic_gate_gamma < 0 < "
                "dynamic_gate_zeta"
            )
        if self.dynamic_gate_temperature_init <= 0 or self.dynamic_gate_temperature_final <= 0:
            raise ValueError("dynamic gate temperatures must be positive")
        self.dynamic_gate_temperature = self.dynamic_gate_temperature_init

    def dynamic_gate_stochastic_active(self):
        """Return whether the current optimizer step should sample gate noise."""
        return bool(
            self.dynamic_enabled
            and not self.dynamic_gate_frozen
            and self.dynamic_gate_training
            and torch.is_grad_enabled()
            and self.dynamic_gate_iteration < self.dynamic_gate_stochastic_until
        )

    def sample_dynamic_gate_noise(self, stochastic=None):
        """Sample detached hard-concrete noise that can be shared by a view batch.

        Each view rebuilds its own straight-through gate graph from this noise.
        Consequently the hard static/dynamic partition is identical across the
        batch while sequential per-view backward passes can release their graphs.
        """
        if not self.dynamic_enabled:
            return None
        if self._dynamic_logit.numel() != self._xyz.shape[0]:
            self._initialize_dynamic_gate(legacy_dynamic=True)
        if stochastic is None:
            stochastic = self.dynamic_gate_stochastic_active()
        if not stochastic:
            return None
        uniform = torch.rand_like(self._dynamic_logit).clamp_(1e-6, 1.0 - 1e-6)
        return (torch.log(uniform) - torch.log1p(-uniform)).detach()

    def sample_dynamic_gate(self, stochastic=None, logistic_noise=None):
        """Return an exact {0,1} gate, with an ST gradient while training."""
        if not self.dynamic_enabled:
            return torch.zeros(
                (self._xyz.shape[0], 1), dtype=self._xyz.dtype, device=self._xyz.device
            )
        if self.dynamic_gate_frozen and self._committed_dynamic_mask.numel() == self._xyz.shape[0]:
            return self._committed_dynamic_mask.reshape(-1, 1).to(dtype=self._xyz.dtype)

        if self._dynamic_logit.numel() != self._xyz.shape[0]:
            self._initialize_dynamic_gate(legacy_dynamic=True)
        if logistic_noise is not None:
            if stochastic is False:
                raise ValueError("logistic_noise requires stochastic=True")
            stochastic = True
        if stochastic is None:
            stochastic = self.dynamic_gate_stochastic_active()

        logits = self._dynamic_logit
        if stochastic:
            if logistic_noise is None:
                logistic_noise = self.sample_dynamic_gate_noise(stochastic=True)
            if logistic_noise.shape != logits.shape:
                raise ValueError(
                    "dynamic gate logistic noise must match logits: "
                    f"got {tuple(logistic_noise.shape)}, expected {tuple(logits.shape)}"
                )
            logistic_noise = logistic_noise.to(device=logits.device, dtype=logits.dtype)
            concrete = torch.sigmoid(
                (logits + logistic_noise) / self.dynamic_gate_temperature
            )
        else:
            concrete = torch.sigmoid(logits)
        relaxed = (
            concrete * (self.dynamic_gate_zeta - self.dynamic_gate_gamma)
            + self.dynamic_gate_gamma
        ).clamp(0.0, 1.0)
        hard = (relaxed >= 0.5).to(relaxed.dtype)
        use_straight_through = (
            self.dynamic_gate_training
            and not self.dynamic_gate_frozen
            and torch.is_grad_enabled()
            and logits.requires_grad
        )
        if use_straight_through:
            gate = hard.detach() - relaxed.detach() + relaxed
        else:
            gate = hard
        # Strong velocity correspondences and explicit promotions are
        # irreversible dynamic evidence.  Allowing the sparsity loss to demote
        # them was the main source of person-shaped static trails.
        if self._forced_dynamic_mask.numel() == gate.shape[0]:
            gate = torch.where(
                self._forced_dynamic_mask[:, None], torch.ones_like(gate), gate
            )
        return gate

    def get_dynamic_mask(self):
        return self.sample_dynamic_gate(stochastic=False).reshape(-1).bool()

    def get_dynamic_gate_regularization(self):
        if not self.dynamic_enabled or self.dynamic_gate_frozen:
            zero = self._xyz.new_tensor(0.0)
            return zero, zero
        # Louizos et al.'s expected L0 probability for a hard-concrete gate.
        expected_l0 = torch.sigmoid(
            self._dynamic_logit
            - self.dynamic_gate_temperature
            * math.log(-self.dynamic_gate_gamma / self.dynamic_gate_zeta)
        )
        probability = torch.sigmoid(self._dynamic_logit)
        free_mask = ~self._forced_dynamic_mask.reshape(-1)
        if not bool(free_mask.any()):
            zero = self._xyz.new_tensor(0.0)
            return zero, zero
        expected_l0 = expected_l0.reshape(-1)[free_mask].mean()
        binary = (probability * (1.0 - probability)).reshape(-1)[free_mask].mean()
        return expected_l0, binary

    def commit_dynamic_partition(self):
        """Permanently materialize the deterministic gate used by the renderer."""
        if not self.dynamic_enabled or self.dynamic_gate_frozen:
            return 0
        mask = self.get_dynamic_mask().detach()
        self._committed_dynamic_mask = mask
        self.dynamic_gate_frozen = True
        self.dynamic_gate_training = False
        if isinstance(self._dynamic_logit, nn.Parameter):
            self._dynamic_logit.requires_grad_(False)
        return int(mask.sum().item())

    @torch.no_grad()
    def update_dynamic_promotions(
        self, iteration, training_args=None, regularization_gradient=None
    ):
        """Promote persistent reconstruction-error outliers to the dynamic set.

        The straight-through gate receives the multi-time photometric gradient.
        A persistently negative reconstruction-gradient component means that
        increasing the dynamic gate would reduce the image loss.  The caller
        supplies the gate regularizer gradient separately so it can be removed
        from this promotion evidence.
        """
        if (
            not self.dynamic_enabled
            or self.dynamic_gate_frozen
            or not isinstance(self._dynamic_logit, nn.Parameter)
            or self._dynamic_logit.grad is None
        ):
            return 0

        decay = float(getattr(
            training_args, "dynamic_promotion_ema_decay", 0.95
        ))
        gate_gradient = self._dynamic_logit.grad.detach()
        if regularization_gradient is not None:
            gate_gradient = gate_gradient - regularization_gradient.detach()
        evidence = torch.relu(-gate_gradient)
        if self._promotion_ema.shape != evidence.shape:
            self._promotion_ema = torch.zeros_like(evidence)
        self._promotion_ema.mul_(decay).add_(evidence, alpha=1.0 - decay)

        warmup = int(getattr(training_args, "dynamic_promotion_warmup", 2_000))
        interval = int(getattr(training_args, "dynamic_promotion_interval", 500))
        if iteration < warmup or interval <= 0 or iteration % interval != 0:
            return 0

        static_mask = ~self.get_dynamic_mask()
        scores = self._promotion_ema.reshape(-1)
        candidates = scores[static_mask]
        if candidates.numel() == 0:
            return 0
        quantile = float(getattr(
            training_args, "dynamic_promotion_quantile", 0.995
        ))
        quantile = min(max(quantile, 0.0), 1.0)
        threshold = max(
            float(torch.quantile(candidates, quantile).item()),
            float(getattr(training_args, "dynamic_promotion_min_score", 1e-7)),
        )
        promote = static_mask & (scores >= threshold)

        max_fraction = float(getattr(
            training_args, "dynamic_promotion_max_fraction", 0.01
        ))
        max_count = max(1, int(math.ceil(self._xyz.shape[0] * max_fraction)))
        promote_indices = torch.nonzero(promote, as_tuple=False).reshape(-1)
        if promote_indices.numel() > max_count:
            _, order = torch.topk(scores[promote_indices], max_count)
            promote_indices = promote_indices[order]
        if promote_indices.numel() == 0:
            return 0

        promotion_logit = float(getattr(
            training_args, "dynamic_promotion_logit", 8.0
        ))
        self._dynamic_logit[ promote_indices, 0 ] = promotion_logit
        if self._forced_dynamic_mask.numel() == self._xyz.shape[0]:
            self._forced_dynamic_mask[promote_indices] = True
        self._promotion_ema[promote_indices] = 0
        return int(promote_indices.numel())

    def mask_committed_static_gradients(self):
        """Freeze only time-varying parameters on committed static rows.

        Static means *independent of time*, not immutable during optimization.
        Its canonical xyz, scale, opacity and appearance must keep fitting the
        multi-view observations after gate commitment.  Freezing those rows,
        the shared appearance networks and their codebooks caused the final
        refinement stage to preserve blurred temporal averages.
        """
        if not self.dynamic_gate_frozen or self._committed_dynamic_mask.numel() == 0:
            return
        static_mask = ~self._committed_dynamic_mask
        for parameter in (
            self._velocity,
            self._acceleration,
            self._time,
            self._duration,
        ):
            if (
                isinstance(parameter, torch.Tensor)
                and parameter.grad is not None
                and parameter.shape[0] == static_mask.shape[0]
            ):
                parameter.grad[static_mask] = 0

    @property
    def get_duration(self):
        _, _, _, duration = self._get_dynamic_attributes()
        duration = torch.exp(duration)
        if self.dynamic_duration_max is not None:
            duration = duration.clamp(min=self.dynamic_duration_min, max=self.dynamic_duration_max)
        else:
            duration = duration.clamp(min=self.dynamic_duration_min)
        return duration
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            return True
        return False

    def construct_dynamic_net(self, training_args=None):
        """Initialize FreeTimeGS-style temporal motion parameters.

        This intentionally keeps the old public method name, but it no longer
        creates an MLP.  Dynamic xyz is represented directly as:
        x(t) = x + velocity * dt + 0.5 * acceleration * dt^2.
        """
        n_points = self._xyz.shape[0]
        if n_points == 0:
            self.dynamic_enabled = True
            return

        device = self._xyz.device
        self._configure_dynamic_gate(training_args)
        time_init = float(getattr(training_args, 'dynamic_time_init', 0.5))
        duration_init = max(float(getattr(training_args, 'dynamic_duration_init', 1.0)), 1e-4)
        self.dynamic_duration_min = max(float(getattr(training_args, 'dynamic_duration_min', 1e-4)), 1e-8)
        duration_max = getattr(training_args, 'dynamic_duration_max', None)
        self.dynamic_duration_max = None if duration_max is None else max(float(duration_max), self.dynamic_duration_min)

        if self._velocity.numel() == 0 or self._velocity.shape[0] != n_points:
            self._velocity = torch.zeros((n_points, 3), dtype=torch.float, device=device)
        else:
            self._velocity = self._velocity.to(device=device, dtype=torch.float)

        if self._acceleration.numel() == 0 or self._acceleration.shape[0] != n_points:
            self._acceleration = torch.zeros((n_points, 3), dtype=torch.float, device=device)
        else:
            self._acceleration = self._acceleration.to(device=device, dtype=torch.float)

        if self._time.numel() == 0 or self._time.shape[0] != n_points:
            if bool(getattr(training_args, 'dynamic_random_time_init', False)):
                self._time = torch.rand((n_points, 1), dtype=torch.float, device=device)
            else:
                self._time = torch.full((n_points, 1), time_init, dtype=torch.float, device=device)
        else:
            self._time = self._time.to(device=device, dtype=torch.float)

        if self._duration.numel() == 0 or self._duration.shape[0] != n_points:
            self._duration = torch.full(
                (n_points, 1),
                np.log(duration_init),
                dtype=torch.float,
                device=device,
            )
        else:
            self._duration = self._duration.to(device=device, dtype=torch.float)

        self._velocity = nn.Parameter(self._velocity.requires_grad_(True))
        self._acceleration = nn.Parameter(self._acceleration.requires_grad_(True))
        self._time = nn.Parameter(self._time.requires_grad_(True))
        self._duration = nn.Parameter(self._duration.requires_grad_(True))
        self._initialize_dynamic_gate(training_args)
        if not self.dynamic_gate_frozen:
            self._dynamic_logit = nn.Parameter(
                self._dynamic_logit.requires_grad_(True)
            )
            self.dynamic_gate_training = training_args is not None
        elif not isinstance(self._dynamic_logit, nn.Parameter):
            self._dynamic_logit = nn.Parameter(
                self._dynamic_logit, requires_grad=False
            )
        self.dynamic_enabled = True

    def get_temporal_opacity(self, time=None, gate=None):
        if not self.dynamic_enabled or time is None:
            return torch.ones((self._xyz.shape[0], 1), dtype=self._xyz.dtype, device=self._xyz.device)

        _, _, canonical_time, _ = self._get_dynamic_attributes()
        duration = self.get_duration
        t = torch.as_tensor(time, dtype=self._xyz.dtype, device=self._xyz.device).reshape(1, 1)
        temporal = torch.exp(-0.5 * ((t - canonical_time) / (duration + 1e-8)) ** 2)
        if gate is None:
            gate = self.sample_dynamic_gate()
        # The hard forward gate makes static opacity exactly time invariant.
        return (1.0 - gate) + gate * temporal

    def get_acceleration_regularization(self, gate=None):
        if not self.dynamic_enabled:
            return self._xyz.new_tensor(0.0)
        _, acceleration, _, _ = self._get_dynamic_attributes()
        if gate is None:
            gate = self.sample_dynamic_gate()
        weight = gate.expand_as(acceleration)
        return (weight * acceleration.square()).sum() / weight.sum().clamp_min(1.0)

    def get_duration_regularization(self, gate=None, soft_max=0.25):
        """Penalize only excessively broad dynamic temporal windows."""
        if not self.dynamic_enabled:
            return self._xyz.new_tensor(0.0)
        if gate is None:
            gate = self.sample_dynamic_gate(stochastic=False)
        weight = gate.detach().reshape(-1, 1)
        duration = self.get_duration
        excess = torch.relu(duration - float(soft_max)).square()
        return (weight * excess).sum() / weight.sum().clamp_min(1.0)

    def get_motion_extent_regularization(self, gate=None, soft_max=0.15):
        """Penalize long visible trails while leaving static widths alone.

        At one temporal standard deviation, the displacement of the quadratic
        trajectory is bounded by |v|*sigma + 0.5*|a|*sigma^2.  Keeping that
        extent compact directly targets duplicated moving edges without
        forcing slow or stationary Gaussians to use unnecessarily short lives.
        """
        if not self.dynamic_enabled:
            return self._xyz.new_tensor(0.0)
        if gate is None:
            gate = self.sample_dynamic_gate(stochastic=False)
        velocity, acceleration, _, _ = self._get_dynamic_attributes()
        duration = self.get_duration
        extent = (
            torch.linalg.vector_norm(velocity, dim=-1, keepdim=True) * duration
            + 0.5
            * torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True)
            * duration.square()
        )
        weight = gate.detach().reshape(-1, 1)
        excess = torch.relu(extent - float(soft_max)).square()
        return (weight * excess).sum() / weight.sum().clamp_min(1.0)

    def constrain_dynamic_parameters(self, training_args=None):
        if not self.dynamic_enabled:
            return

        duration_min = self.dynamic_duration_min
        duration_max = self.dynamic_duration_max
        # A zero/missing limit means that initialized velocities are preserved.
        # This is important because a per-frame velocity is scaled into the
        # normalized scene clock by the dataset loader and can legitimately be
        # much larger than one.
        velocity_max = float(getattr(training_args, 'dynamic_velocity_max', 0.0) or 0.0)
        acceleration_max = getattr(training_args, 'dynamic_acceleration_max', None)
        if training_args is not None:
            duration_min = max(float(getattr(training_args, 'dynamic_duration_min', duration_min)), 1e-8)
            arg_duration_max = getattr(training_args, 'dynamic_duration_max', duration_max)
            duration_max = None if arg_duration_max is None else max(float(arg_duration_max), duration_min)

        with torch.no_grad():
            log_duration_min = float(np.log(duration_min))
            log_duration_max = None if duration_max is None else float(np.log(duration_max))
            if self.vq_enabled and hasattr(self, "dynamic_codes") and len(self.dynamic_codes) > 0:
                dynamic_attrs = torch.cat([code.detach() for code in self.dynamic_codes], dim=-1)
                if dynamic_attrs.shape[1] >= 8 and getattr(training_args, 'slice_dynamic', 1) == 1:
                    if velocity_max > 0:
                        decoded_velocity, _, _, _ = self._get_dynamic_attributes()
                        decoded_norm = torch.linalg.vector_norm(decoded_velocity, dim=-1, keepdim=True)
                        point_scale = torch.clamp(
                            velocity_max / decoded_norm.clamp_min(1e-12),
                            max=1.0,
                        ).squeeze(-1)
                        clipped_count = int((point_scale < 1.0).sum().item())
                        if clipped_count:
                            # Each velocity component has its own scalar
                            # codebook.  Use the smallest scale requested by
                            # any point sharing a code so every decoded vector
                            # respects the Euclidean cap.
                            for code, indices in zip(
                                self.dynamic_codes[:3],
                                self.dynamic_indices[:3],
                            ):
                                code_scale = torch.ones(
                                    code.shape[0],
                                    dtype=point_scale.dtype,
                                    device=point_scale.device,
                                )
                                code_scale.scatter_reduce_(
                                    0,
                                    indices.reshape(-1),
                                    point_scale,
                                    reduce="amin",
                                    include_self=True,
                                )
                                code.mul_(code_scale[:, None])
                            print(
                                f"Clipped {clipped_count:,}/{decoded_velocity.shape[0]:,} "
                                f"quantized velocities to norm <= {velocity_max:g}"
                            )
                    if acceleration_max is not None and acceleration_max > 0:
                        for i in range(3, 6):
                            self.dynamic_codes[i].clamp_(min=-float(acceleration_max), max=float(acceleration_max))
                    self.dynamic_codes[6].clamp_(min=0.0, max=1.0)
                    if log_duration_max is None:
                        self.dynamic_codes[7].clamp_(min=log_duration_min)
                    else:
                        self.dynamic_codes[7].clamp_(min=log_duration_min, max=log_duration_max)
                return

            if self._time.numel() > 0:
                self._time.clamp_(min=0.0, max=1.0)
            if self._duration.numel() > 0:
                if log_duration_max is None:
                    self._duration.clamp_(min=log_duration_min)
                else:
                    self._duration.clamp_(min=log_duration_min, max=log_duration_max)
            if velocity_max > 0 and self._velocity.numel() > 0:
                velocity_norm = torch.linalg.vector_norm(self._velocity, dim=-1, keepdim=True)
                velocity_scale = torch.clamp(velocity_max / velocity_norm.clamp_min(1e-12), max=1.0)
                clipped_count = int((velocity_scale < 1.0).sum().item())
                if clipped_count:
                    self._velocity.mul_(velocity_scale)
                    print(
                        f"Clipped {clipped_count:,}/{self._velocity.shape[0]:,} velocities "
                        f"to norm <= {velocity_max:g}"
                    )
            if acceleration_max is not None and acceleration_max > 0 and self._acceleration.numel() > 0:
                self._acceleration.clamp_(min=-float(acceleration_max), max=float(acceleration_max))

    def get_deformed_xyz(self, time=None, return_offset=False, gate=None):
        if not self.dynamic_enabled or time is None:
            if return_offset:
                return self._xyz, None
            return self._xyz

        velocity, acceleration, canonical_time, _ = self._get_dynamic_attributes()
        t = torch.as_tensor(time, dtype=self._xyz.dtype, device=self._xyz.device).reshape(1, 1)
        dt = t - canonical_time
        offset = velocity * dt + 0.5 * acceleration * dt.square()
        if gate is None:
            gate = self.sample_dynamic_gate()
        offset = gate * offset
        deformed = self._xyz + offset
        if return_offset:
            return deformed, offset
        return deformed

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"Point-cloud positions must be a non-empty Nx3 array, got {points.shape}")
        if colors.shape != points.shape:
            raise ValueError(f"Point-cloud colors must have shape {points.shape}, got {colors.shape}")
        if not np.isfinite(points).all() or not np.isfinite(colors).all():
            raise ValueError("Point-cloud positions and colors must contain only finite values")

        fused_point_cloud = torch.tensor(points).float().cuda()
        fused_color = RGB2SH(torch.tensor(colors).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(points).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        motion_values = {
            "velocities": getattr(pcd, "velocities", None),
            "times": getattr(pcd, "times", None),
            "durations": getattr(pcd, "durations", None),
        }
        supplied_motion_fields = [name for name, value in motion_values.items() if value is not None]
        if supplied_motion_fields and len(supplied_motion_fields) != len(motion_values):
            missing = sorted(set(motion_values) - set(supplied_motion_fields))
            raise ValueError(
                "Velocity initialization is incomplete: supplied "
                f"{sorted(supplied_motion_fields)}, missing {missing}"
            )

        if supplied_motion_fields:
            velocities = np.asarray(motion_values["velocities"], dtype=np.float32)
            times = np.asarray(motion_values["times"], dtype=np.float32)
            durations = np.asarray(motion_values["durations"], dtype=np.float32)
            if times.ndim == 1:
                times = times[:, None]
            if durations.ndim == 1:
                durations = durations[:, None]

            expected_rows = len(points)
            expected_shapes = {
                "velocities": (expected_rows, 3),
                "times": (expected_rows, 1),
                "durations": (expected_rows, 1),
            }
            for name, values in (
                ("velocities", velocities),
                ("times", times),
                ("durations", durations),
            ):
                if values.shape != expected_shapes[name]:
                    raise ValueError(
                        f"Velocity initialization {name} must have shape "
                        f"{expected_shapes[name]}, got {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise ValueError(f"Velocity initialization {name} contains non-finite values")
            if np.any(durations <= 0):
                raise ValueError("Velocity initialization durations must all be positive")
            if np.any(times < 0) or np.any(times > 1):
                raise ValueError("Velocity initialization times must lie in the normalized [0, 1] range")

            self._velocity = torch.as_tensor(velocities, dtype=torch.float, device="cuda")
            self._acceleration = torch.zeros_like(self._velocity)
            self._time = torch.as_tensor(times, dtype=torch.float, device="cuda")
            duration_tensor = torch.as_tensor(durations, dtype=torch.float, device="cuda")
            self._duration = torch.log(duration_tensor)
            self._dynamic_logit = torch.empty(0, device=fused_point_cloud.device)
            self._forced_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=fused_point_cloud.device
            )
            self._committed_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=fused_point_cloud.device
            )
            self._promotion_ema = torch.empty(0, device=fused_point_cloud.device)
            self.dynamic_gate_frozen = False
            self.dynamic_enabled = True

            speeds = np.linalg.norm(velocities, axis=1)
            q50, q90, q99 = np.quantile(speeds, (0.5, 0.9, 0.99))
            print(
                "Loaded velocity initialization: "
                f"speed q50={q50:.6g}, q90={q90:.6g}, "
                f"q99={q99:.6g}, max={speeds.max():.6g}"
            )
        else:
            self._velocity = torch.empty(0, device=fused_point_cloud.device)
            self._acceleration = torch.empty(0, device=fused_point_cloud.device)
            self._time = torch.empty(0, device=fused_point_cloud.device)
            self._duration = torch.empty(0, device=fused_point_cloud.device)
            self._dynamic_logit = torch.empty(0, device=fused_point_cloud.device)
            self._forced_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=fused_point_cloud.device
            )
            self._committed_dynamic_mask = torch.empty(
                0, dtype=torch.bool, device=fused_point_cloud.device
            )
            self._promotion_ema = torch.empty(0, device=fused_point_cloud.device)
            self.dynamic_gate_frozen = False
            self.dynamic_enabled = False

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if getattr(training_args, 'dynamic', False) or self.dynamic_enabled:
            self.construct_dynamic_net(training_args)
        if self.dynamic_enabled:
            with torch.no_grad():
                velocity, _, _, _ = self._get_dynamic_attributes()
                speeds = torch.linalg.vector_norm(velocity.detach().float(), dim=-1)
                if speeds.numel() > 0:
                    q50, q90, q99 = torch.quantile(
                        speeds,
                        torch.tensor((0.5, 0.9, 0.99), device=speeds.device),
                    ).cpu().tolist()
                    
            # Apply a velocity limit only when the caller explicitly supplies
            # a positive --dynamic_velocity_max value.
            self.constrain_dynamic_parameters(training_args)

        if self.net_enabled:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._features_rot], 'lr': getattr(training_args, 'rot_feature_lr', 0.0025), "name": "f_rot"},
                {'params': [self._features_static], 'lr': training_args.feature_lr, "name": "f_static"},
                {'params': [self._features_view], 'lr': training_args.feature_lr, "name": "f_view"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.lowfeature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.highfeature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]

        if self.dynamic_enabled:
            self.dynamic_temporal_lr_decay_start = int(getattr(
                training_args, "dynamic_temporal_lr_decay_start", 25_000
            ))
            self.dynamic_temporal_lr_freeze_iter = int(getattr(
                training_args, "dynamic_temporal_lr_freeze_iter", 30_000
            ))
            # Validate the schedule when training is configured, rather than
            # failing much later when the decay boundary is reached.
            cosine_decay_to_zero(
                0,
                self.dynamic_temporal_lr_decay_start,
                self.dynamic_temporal_lr_freeze_iter,
            )
            self.dynamic_temporal_base_lrs = {
                "acceleration": float(getattr(training_args, 'dynamic_acceleration_lr', getattr(training_args, 'dynamic_lr', 1e-3))),
                "time": float(getattr(training_args, 'dynamic_time_lr', getattr(training_args, 'dynamic_lr', 1e-3))),
                "duration": float(getattr(training_args, 'dynamic_duration_lr', getattr(training_args, 'dynamic_lr', 1e-3))),
            }
            l.extend([
                {'params': [self._velocity], 'lr': getattr(training_args, 'dynamic_velocity_lr', getattr(training_args, 'dynamic_lr', 1e-3)), "name": "velocity"},
                {'params': [self._acceleration], 'lr': self.dynamic_temporal_base_lrs["acceleration"], "name": "acceleration"},
                {'params': [self._time], 'lr': self.dynamic_temporal_base_lrs["time"], "name": "time"},
                {'params': [self._duration], 'lr': self.dynamic_temporal_base_lrs["duration"], "name": "duration"},
            ])
            l.append({
                'params': [self._dynamic_logit],
                'lr': (
                    0.0 if self.dynamic_gate_frozen else
                    float(getattr(training_args, 'dynamic_gate_lr', 5e-3))
                ),
                "name": "dynamic_gate",
            })

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = SparseGaussianAdam(l + sh_l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if self.dynamic_enabled:
            self.velocity_scheduler_args = get_expon_lr_func(
                lr_init=getattr(training_args, 'dynamic_velocity_lr', getattr(training_args, 'dynamic_lr', 1e-3)),
                lr_final=getattr(training_args, 'dynamic_velocity_lr_final', getattr(training_args, 'dynamic_velocity_lr', getattr(training_args, 'dynamic_lr', 1e-3))),
                lr_delay_mult=1.0,
                max_steps=training_args.position_lr_max_steps,
            )
        else:
            self.velocity_scheduler_args = None

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        xyz_lr = None
        self.dynamic_gate_iteration = int(iteration)
        if self.dynamic_enabled and not self.dynamic_gate_frozen:
            anneal_start = self.dynamic_gate_anneal_start
            anneal_end = max(self.dynamic_gate_freeze_iter, anneal_start + 1)
            progress = min(max((iteration - anneal_start) / (anneal_end - anneal_start), 0.0), 1.0)
            # Geometric interpolation avoids spending most of training at the
            # high-temperature end of a wide annealing range.
            self.dynamic_gate_temperature = (
                self.dynamic_gate_temperature_init
                * (self.dynamic_gate_temperature_final / self.dynamic_gate_temperature_init) ** progress
            )
        temporal_lr_scale = cosine_decay_to_zero(
            iteration,
            self.dynamic_temporal_lr_decay_start,
            self.dynamic_temporal_lr_freeze_iter,
        ) if self.dynamic_enabled else 1.0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                xyz_lr = lr
            elif param_group["name"] == "velocity" and self.velocity_scheduler_args is not None:
                param_group['lr'] = self.velocity_scheduler_args(iteration)
            elif param_group["name"] in self.dynamic_temporal_base_lrs:
                param_group['lr'] = (
                    self.dynamic_temporal_base_lrs[param_group["name"]]
                    * temporal_lr_scale
                )
        return xyz_lr


    def construct_list_of_attributes(self, features_dc=None, features_rest=None, scaling=None, rotation=None):
        features_dc = self._features_dc if features_dc is None else features_dc
        features_rest = self._features_rest if features_rest is None else features_rest
        scaling = self._scaling if scaling is None else scaling
        rotation = self._rotation if rotation is None else rotation

        l = ['x', 'y', 'z',]
        # All channels except the 3 DC
        for i in range(features_dc.shape[1]*features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(features_rest.shape[1]*features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.dynamic_enabled:
            for i in range(self._velocity.shape[1]):
                l.append('velocity_{}'.format(i))
            for i in range(self._acceleration.shape[1]):
                l.append('acceleration_{}'.format(i))
            l.append('time')
            l.append('duration')
            l.append('dynamic_logit')
            l.append('dynamic_gate')
        return l

    def _get_contract_aabb(self, device):
        if self._contract_aabb is None or self._contract_aabb.device != device:
            self._contract_aabb = torch.tensor(
                [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                dtype=torch.float32,
                device=device,
            )
        return self._contract_aabb

    def _contract_xyz(self, xyz):
        return self.contract_to_unisphere(xyz, self._get_contract_aabb(xyz.device))

    def _attributes_for_ply(self):
        if not self.net_enabled:
            return (
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._opacity,
                self._scaling,
                self._rotation,
            )

        with torch.no_grad():
            cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
            if self.vq_enabled:
                app_feature = self.get_svq_appearance
                rot_feature = self.get_svq_rot_feature
                space_app = app_feature[:, 0:3]
                view_app = app_feature[:, 3:6]
            else:
                rot_feature = self._features_rot
                space_app = self._features_static
                view_app = self._features_view

            space_feature = torch.cat([cont_feature, space_app], dim=-1)
            view_feature = torch.cat([cont_feature, view_app], dim=-1)
            f_rest = self.mlp_view(view_feature).reshape(-1, self.max_sh_rest, 3).float()
            f_dc = self.mlp_dc(space_feature).reshape(-1, 1, 3).float()
            opacities = self.mlp_opacity(space_feature).float()

            rot_input = torch.cat([cont_feature, rot_feature], dim=-1)
            rotation = torch.nn.functional.normalize(self.mlp_rotation(rot_input).float(), dim=-1)

            if self.shoffset_enabled and hasattr(self, "shs_nn"):
                shs = torch.cat([f_dc, f_rest], dim=1)
                sh_offset = self.get_features_offset(shs, self.opacity_activation(opacities), rotation)
                f_dc = f_dc + sh_offset[:, 0:1]
                f_rest = f_rest + sh_offset[:, 1:]

        return self._xyz, f_dc, f_rest, opacities, self._scaling, rotation

    def optimizer_step(self, iteration):
        ''' An optimization schdeuler. The goal is similar to the sparse Adam of taming 3dgs.'''
        if iteration <= 15000:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none = True)
            if iteration % 16 == 0:
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        elif iteration <= 20000:
            if iteration % 32 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)
        else:
            if iteration % 64 ==0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none = True)
                self.shoptimizer.step()
                self.shoptimizer.zero_grad(set_to_none = True)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz_t, f_dc_t, f_rest_t, opacities_t, scale_t, rotation_t = self._attributes_for_ply()
        row_counts = {
            "xyz": xyz_t.shape[0],
            "f_dc": f_dc_t.shape[0],
            "f_rest": f_rest_t.shape[0],
            "opacity": opacities_t.shape[0],
            "scale": scale_t.shape[0],
            "rotation": rotation_t.shape[0],
        }
        if self.dynamic_enabled:
            row_counts["velocity"] = self._velocity.shape[0]
            row_counts["acceleration"] = self._acceleration.shape[0]
            row_counts["time"] = self._time.shape[0]
            row_counts["duration"] = self._duration.shape[0]
            row_counts["dynamic_logit"] = self._dynamic_logit.shape[0]
        if len(set(row_counts.values())) != 1:
            raise ValueError(f"Cannot save PLY with inconsistent Gaussian attribute counts: {row_counts}")

        xyz = xyz_t.detach().cpu().numpy()
        f_dc = f_dc_t.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = f_rest_t.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacities_t.detach().cpu().numpy()
        scale = scale_t.detach().cpu().numpy()
        rotation = rotation_t.detach().cpu().numpy()
        dynamic_attributes = []
        if self.dynamic_enabled:
            dynamic_attributes.extend([
                self._velocity.detach().cpu().numpy(),
                self._acceleration.detach().cpu().numpy(),
                self._time.detach().cpu().numpy(),
                self.get_duration.detach().cpu().numpy(),
                self._dynamic_logit.detach().cpu().numpy(),
                self.get_dynamic_mask().to(dtype=torch.float32).reshape(-1, 1).cpu().numpy(),
            ])

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(f_dc_t, f_rest_t, scale_t, rotation_t)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, f_dc, f_rest, opacities, scale, rotation, *dynamic_attributes), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) % 3 == 0
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, len(extra_f_names) // 3))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        property_names = {p.name for p in plydata.elements[0].properties}
        has_dynamic = {"velocity_0", "velocity_1", "velocity_2", "time", "duration"}.issubset(property_names)
        has_acceleration = {"acceleration_0", "acceleration_1", "acceleration_2"}.issubset(property_names)
        if has_dynamic:
            velocities = np.stack((np.asarray(plydata.elements[0]["velocity_0"]),
                                   np.asarray(plydata.elements[0]["velocity_1"]),
                                   np.asarray(plydata.elements[0]["velocity_2"])), axis=1)
            if has_acceleration:
                accelerations = np.stack((np.asarray(plydata.elements[0]["acceleration_0"]),
                                          np.asarray(plydata.elements[0]["acceleration_1"]),
                                          np.asarray(plydata.elements[0]["acceleration_2"])), axis=1)
            else:
                accelerations = np.zeros_like(velocities)
            times = np.asarray(plydata.elements[0]["time"])[..., np.newaxis]
            durations = np.asarray(plydata.elements[0]["duration"])[..., np.newaxis]
            if "dynamic_logit" in property_names:
                dynamic_logits = np.asarray(
                    plydata.elements[0]["dynamic_logit"]
                )[..., np.newaxis]
            else:
                dynamic_logits = np.full(
                    (xyz.shape[0], 1),
                    self._probability_to_logit(0.95),
                    dtype=np.float32,
                )
            if "dynamic_gate" in property_names:
                committed_dynamic_mask = np.asarray(
                    plydata.elements[0]["dynamic_gate"]
                ) >= 0.5
            else:
                committed_dynamic_mask = None

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        if has_dynamic:
            self._velocity = nn.Parameter(torch.tensor(velocities, dtype=torch.float, device="cuda").requires_grad_(True))
            self._acceleration = nn.Parameter(torch.tensor(accelerations, dtype=torch.float, device="cuda").requires_grad_(True))
            self._time = nn.Parameter(torch.tensor(times, dtype=torch.float, device="cuda").requires_grad_(True))
            duration_tensor = torch.tensor(durations, dtype=torch.float, device="cuda").clamp(min=1e-4)
            self._duration = nn.Parameter(torch.log(duration_tensor).requires_grad_(True))
            self._dynamic_logit = nn.Parameter(
                torch.tensor(dynamic_logits, dtype=torch.float, device="cuda").requires_grad_(True)
            )
            if committed_dynamic_mask is not None:
                self._committed_dynamic_mask = torch.tensor(
                    committed_dynamic_mask, dtype=torch.bool, device="cuda"
                )
                self._forced_dynamic_mask = self._committed_dynamic_mask.clone()
                self.dynamic_gate_frozen = True
                self._dynamic_logit.requires_grad_(False)
            else:
                self._forced_dynamic_mask = torch.empty(
                    0, dtype=torch.bool, device="cuda"
                )
                self._committed_dynamic_mask = torch.empty(0, dtype=torch.bool, device="cuda")
                self.dynamic_gate_frozen = False
            self._promotion_ema = torch.zeros_like(self._dynamic_logit)
            self.dynamic_enabled = True
        else:
            self._velocity = torch.empty(0)
            self._acceleration = torch.empty(0)
            self._time = torch.empty(0)
            self._duration = torch.empty(0)
            self._dynamic_logit = torch.empty(0)
            self._forced_dynamic_mask = torch.empty(0, dtype=torch.bool)
            self._committed_dynamic_mask = torch.empty(0, dtype=torch.bool)
            self._promotion_ema = torch.empty(0)
            self.dynamic_gate_frozen = False
            self.dynamic_enabled = False

        loaded_sh_coeffs = features_extra.shape[2] + 1
        loaded_sh_degree = int(np.sqrt(loaded_sh_coeffs) - 1)
        if (loaded_sh_degree + 1) ** 2 == loaded_sh_coeffs:
            self.active_sh_degree = min(self.max_sh_degree, loaded_sh_degree)
        else:
            self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask

        # SVQ codebooks are shared, but every index array has one entry per
        # Gaussian. Final pruning may still run after the 3D SVQ stage, so keep
        # those row-wise assignments aligned with the tensors pruned below.
        pruned_svq_indices = {}
        if self.vq_enabled:
            for index_group_name in (
                "opacity_indices",
                "scale_indices",
                "rotation_indices",
                "appearance_indices",
                "dynamic_indices",
            ):
                index_group = getattr(self, index_group_name, [])
                for index in index_group:
                    if index.numel() != valid_points_mask.numel():
                        raise RuntimeError(
                            f"Cannot prune {index_group_name}: index rows "
                            f"{index.numel()} do not match Gaussian rows "
                            f"{valid_points_mask.numel()}"
                        )
                pruned_svq_indices[index_group_name] = [
                    index[valid_points_mask] for index in index_group
                ]

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        self._scaling = optimizable_tensors["scaling"]

        if self.dynamic_enabled:
            self._velocity = optimizable_tensors["velocity"]
            self._acceleration = optimizable_tensors["acceleration"]
            self._time = optimizable_tensors["time"]
            self._duration = optimizable_tensors["duration"]
            if "dynamic_gate" in optimizable_tensors:
                self._dynamic_logit = optimizable_tensors["dynamic_gate"]
            else:
                self._dynamic_logit = self._dynamic_logit[valid_points_mask]
            if self.dynamic_gate_frozen and isinstance(self._dynamic_logit, nn.Parameter):
                self._dynamic_logit.requires_grad_(False)
            self._promotion_ema = self._promotion_ema[valid_points_mask]
            if self._forced_dynamic_mask.numel() == valid_points_mask.shape[0]:
                self._forced_dynamic_mask = self._forced_dynamic_mask[valid_points_mask]
            if self._committed_dynamic_mask.numel() == valid_points_mask.shape[0]:
                self._committed_dynamic_mask = self._committed_dynamic_mask[valid_points_mask]

        if self.net_enabled:
            self._features_rot = optimizable_tensors["f_rot"]
            self._features_static = optimizable_tensors["f_static"]
            self._features_view = optimizable_tensors["f_view"]
            self._rotation = self._rotation[valid_points_mask]
        else:
            self._rotation = optimizable_tensors["rotation"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        if self.tmp_radii is not None:
            self.tmp_radii = self.tmp_radii[valid_points_mask]

        for index_group_name, index_group in pruned_svq_indices.items():
            setattr(self, index_group_name, index_group)

    def _dynamic_tensors_for_selected(self, selected_pts_mask, repeat=1):
        if not self.dynamic_enabled:
            return {}
        if self._committed_dynamic_mask.numel() == self._xyz.shape[0]:
            committed = self._committed_dynamic_mask
        else:
            committed = self.get_dynamic_mask().detach()
        return {
            "new_velocity": self._velocity[selected_pts_mask].repeat(repeat, 1),
            "new_acceleration": self._acceleration[selected_pts_mask].repeat(repeat, 1),
            "new_time": self._time[selected_pts_mask].repeat(repeat, 1),
            "new_duration": self._duration[selected_pts_mask].repeat(repeat, 1),
            "new_dynamic_logit": self._dynamic_logit[selected_pts_mask].repeat(repeat, 1),
            "new_forced_dynamic_mask": self._forced_dynamic_mask[selected_pts_mask].repeat(repeat),
            "new_committed_dynamic_mask": committed[selected_pts_mask].repeat(repeat),
            "new_promotion_ema": self._promotion_ema[selected_pts_mask].repeat(repeat, 1),
        }

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        optimizers = [self.optimizer]
        if self.shoptimizer: optimizers.append(self.shoptimizer)

        for opt in optimizers:
            for group in opt.param_groups:
                assert len(group["params"]) == 1
                extension_tensor = tensors_dict[group["name"]]
                
                stored_state = opt.state.get(group['params'][0], None)
                if stored_state is not None:

                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                    del opt.state[group['params'][0]]
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    opt.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_static, new_view, new_features_rot=None, new_velocity=None, new_acceleration=None, new_time=None, new_duration=None, new_dynamic_logit=None, new_forced_dynamic_mask=None, new_committed_dynamic_mask=None, new_promotion_ema=None):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "f_rest": new_features_rest,
        "f_static": new_static,
        "f_view": new_view}
        if new_features_rot is not None:
            d["f_rot"] = new_features_rot
        if self.dynamic_enabled:
            d["velocity"] = new_velocity
            d["acceleration"] = new_acceleration
            d["time"] = new_time
            d["duration"] = new_duration
            d["dynamic_gate"] = new_dynamic_logit

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._scaling = optimizable_tensors["scaling"]
        if self.dynamic_enabled:
            self._velocity = optimizable_tensors["velocity"]
            self._acceleration = optimizable_tensors["acceleration"]
            self._time = optimizable_tensors["time"]
            self._duration = optimizable_tensors["duration"]
            if "dynamic_gate" in optimizable_tensors:
                self._dynamic_logit = optimizable_tensors["dynamic_gate"]
            else:
                self._dynamic_logit = torch.cat(
                    (self._dynamic_logit, new_dynamic_logit), dim=0
                )
            if self.dynamic_gate_frozen and isinstance(self._dynamic_logit, nn.Parameter):
                self._dynamic_logit.requires_grad_(False)
            self._promotion_ema = torch.cat(
                (self._promotion_ema, new_promotion_ema), dim=0
            )
            self._forced_dynamic_mask = torch.cat((
                self._forced_dynamic_mask,
                new_forced_dynamic_mask.to(dtype=torch.bool),
            ))
            if self.dynamic_gate_frozen:
                if new_committed_dynamic_mask is None:
                    new_committed_dynamic_mask = new_dynamic_logit.reshape(-1) >= 0
                self._committed_dynamic_mask = torch.cat((
                    self._committed_dynamic_mask,
                    new_committed_dynamic_mask.to(dtype=torch.bool),
                ))
        if not self.net_enabled:
            self._rotation = optimizable_tensors["rotation"]

        if self.net_enabled:
            self._features_rot = optimizable_tensors["f_rot"]
            self._features_static = optimizable_tensors["f_static"]
            self._features_view = optimizable_tensors["f_view"]
        else:
            self._features_dc = optimizable_tensors["f_dc"]
            self._opacity = optimizable_tensors["opacity"]

            self._features_rest = optimizable_tensors["f_rest"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")  # abs
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split_mobilegs2(self, mask, split_distance=0.45, opacity_reduction=0.6):
        """Split selected Gaussians into two children along their longest local axis.

        This follows Improved-GS LAS: children are placed symmetrically along
        the rotated longest axis and their scales are adjusted to cover the
        parent footprint. Directly parameterized opacity is reduced for both
        children; neural-stage opacity remains encoded by the copied features.
        """
        split_distance = float(split_distance)
        opacity_reduction = float(opacity_reduction)
        if not 0.0 <= split_distance < 1.0:
            raise ValueError(f"split_distance must be in [0, 1), got {split_distance}")
        if not 0.0 < opacity_reduction <= 1.0:
            raise ValueError(f"opacity_reduction must be in (0, 1], got {opacity_reduction}")

        n_init_points = self.get_xyz.shape[0]

        selected_pts_mask = torch.zeros((n_init_points), dtype=bool, device="cuda")
        selected_pts_mask[:mask.shape[0]] = mask

        stds = self.get_scaling[selected_pts_mask]
        max_values, max_indices = torch.max(stds, dim=1, keepdim=True)
        axis_mask = torch.zeros_like(stds, dtype=torch.bool).scatter(1, max_indices, True)
        axis_offsets = stds * axis_mask * (3.0 * split_distance)
        axis_offsets = torch.cat((axis_offsets, -axis_offsets), dim=0)

        rate_w = max(1.0 - split_distance, 1e-6)
        rate_h = math.sqrt(max(1.0 - split_distance * split_distance, 1e-6))
        child_scales = stds.scatter(1, max_indices, max_values * rate_w / rate_h)
        child_scales = child_scales.repeat(2, 1) * rate_h
        new_scaling = self.scaling_inverse_activation(child_scales)

        if self.net_enabled:
            with torch.no_grad():
                cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
                decoded_rots = torch.nn.functional.normalize(self.mlp_rotation(torch.cat([cont_feature, self._features_rot], dim=-1)).float(), dim=-1)
            rots = build_rotation(decoded_rots[selected_pts_mask]).repeat(2, 1, 1)
            new_xyz = torch.bmm(rots, axis_offsets.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
            new_features_rot = self._features_rot[selected_pts_mask].repeat(2, 1)
            new_rotation = decoded_rots[selected_pts_mask].repeat(2, 1)

            new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(2)

            new_static = self._features_static[selected_pts_mask].repeat(2, 1)
            new_view = self._features_view[selected_pts_mask].repeat(2, 1)
            dynamic_tensors = self._dynamic_tensors_for_selected(selected_pts_mask, 2)
            self.densification_postfix(new_xyz, None, None, None, new_scaling, None, new_tmp_radii, new_static, new_view, new_features_rot, **dynamic_tensors)
            self._rotation = torch.cat([self._rotation, new_rotation])
        else:
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(2, 1, 1)
            new_xyz = torch.bmm(rots, axis_offsets.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(2, 1)
            new_rotation = self._rotation[selected_pts_mask].repeat(2, 1)

            new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(2)

            new_features_dc = self._features_dc[selected_pts_mask].repeat(2, 1, 1)
            new_opacity = self.inverse_opacity_activation(
                self.get_opacity[selected_pts_mask] * opacity_reduction
            ).repeat(2, 1)

            new_features_rest = self._features_rest[selected_pts_mask].repeat(2, 1, 1)

            dynamic_tensors = self._dynamic_tensors_for_selected(selected_pts_mask, 2)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, None, None, **dynamic_tensors)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(2 * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone_mobilegs2(self, selected_pts_mask):

        new_xyz = self._xyz[selected_pts_mask]

        new_scaling = self._scaling[selected_pts_mask]
        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        if self.net_enabled:
            new_features_rot = self._features_rot[selected_pts_mask]
            with torch.no_grad():
                cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
                decoded_rots = torch.nn.functional.normalize(self.mlp_rotation(torch.cat([cont_feature, self._features_rot], dim=-1)).float(), dim=-1)
            new_rotation = decoded_rots[selected_pts_mask]

            new_static = self._features_static[selected_pts_mask]
            new_view = self._features_view[selected_pts_mask]
            dynamic_tensors = self._dynamic_tensors_for_selected(selected_pts_mask)
            self.densification_postfix(new_xyz, None, None, None, new_scaling, None, new_tmp_radii, new_static, new_view, new_features_rot, **dynamic_tensors)
            self._rotation = torch.cat([self._rotation, new_rotation])
        else:
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            dynamic_tensors = self._dynamic_tensors_for_selected(selected_pts_mask)
            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, None, None, **dynamic_tensors)

    def densify_and_prune_mobilegs2(self, max_screen_size, min_opacity, extent, radii, args, importance_score = None, pruning_score = None, importance_quantile=None):
        
      
        grad_vars = self.xyz_gradient_accum / self.denom
        grad_vars[grad_vars.isnan()] = 0.0
        self.tmp_radii = radii

        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0

        grad_qualifiers = torch.where(torch.norm(grad_vars, dim=-1) >= args.grad_thresh, True, False)
        grad_qualifiers_abs = torch.where(torch.norm(grads_abs, dim=-1) >= args.grad_abs_thresh, True, False)
        clone_qualifiers = torch.max(self.get_scaling, dim=1).values <= args.dense*extent
        split_qualifiers = torch.max(self.get_scaling, dim=1).values > args.dense*extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers_abs)

  
        metric_mask = importance_score > torch.quantile(importance_score, importance_quantile)
        
        clone_mask = torch.logical_and(metric_mask, all_clones)
        split_mask = torch.logical_and(metric_mask, all_splits)

        self.densify_and_clone_mobilegs2(clone_mask)
        self.densify_and_split_mobilegs2(
            split_mask,
            split_distance=getattr(args, "split_distance", 0.45),
            opacity_reduction=getattr(args, "opacity_reduction", 0.6),
        )

        if self.net_enabled:
            cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
            if self.vq_enabled:
                app_feature = self.get_svq_appearance
                space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
            else:
                space_feature = torch.cat([cont_feature, self._features_static],dim=-1)
            opacity = self.opacity_activation(self.mlp_opacity(space_feature).float())

        else:
            opacity = self.get_opacity

        prune_mask = (opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        scores = 1 - pruning_score 
        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        # The budget is not necessary for our method.
        if remove_budget:
            n_init_points = self.get_xyz.shape[0]
            padded_importance = torch.zeros((n_init_points), dtype=torch.float32, device=scores.device)
            padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
            selected_pts_mask = torch.zeros_like(padded_importance, dtype=bool)
            sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
            selected_pts_mask[sampled_indices] = True
            final_prune = torch.logical_and(prune_mask, selected_pts_mask)
            self.prune_points(final_prune)
        
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.8))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        if not self.net_enabled:
            self._opacity = optimizable_tensors["opacity"]
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        xy_gradient = torch.norm(
            viewspace_point_tensor.grad[:, :2], dim=-1, keepdim=True
        )
        abs_gradient = torch.norm(
            viewspace_point_tensor.grad[:, 2:], dim=-1, keepdim=True
        )
        self.add_densification_stats_batch(
            xy_gradient, abs_gradient, update_filter
        )

    def add_densification_stats_batch(
        self, xy_gradient, abs_gradient, update_filter
    ):
        """Accumulate one optimizer batch of pre-aggregated screen gradients."""
        if update_filter.dtype != torch.bool:
            indices = update_filter.reshape(-1).long()
            mask = torch.zeros(
                self.xyz_gradient_accum.shape[0],
                dtype=torch.bool,
                device=self.xyz_gradient_accum.device,
            )
            mask[indices] = True
            update_filter = mask
        else:
            update_filter = update_filter.reshape(-1)
        expected_shape = self.xyz_gradient_accum.shape
        if xy_gradient.shape != expected_shape or abs_gradient.shape != expected_shape:
            raise ValueError(
                "Pre-aggregated densification gradients must have shape "
                f"{tuple(expected_shape)}, got xy={tuple(xy_gradient.shape)}, "
                f"abs={tuple(abs_gradient.shape)}"
            )
        self.xyz_gradient_accum[update_filter] += xy_gradient[update_filter]
        self.xyz_gradient_accum_abs[update_filter] += abs_gradient[update_filter]
        self.denom[update_filter] += 1

    def final_prune_mobilegs2(self, min_opacity, pruning_score = None, pruning_quantile=None):
        """Final-stage pruning: remove Gaussians based on opacity and multi-view consistency.
        In the final stage we remove Gaussians that have low opacity or that are flagged by
        our multi-view reconstruction consistency metric (provided as `pruning_score`)."""


        if self.net_enabled == False:
            opacity = self.get_opacity
        else:
            cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
            if self.vq_enabled:
                app_feature = self.get_svq_appearance
                space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
            else:
                space_feature = torch.cat([cont_feature, self._features_static],dim=-1)

            opacity = self.opacity_activation(self.mlp_opacity(space_feature).float())                


        prune_mask = (opacity < min_opacity).squeeze() 
        scores_mask = pruning_score > torch.quantile(pruning_score, pruning_quantile)
        final_prune = torch.logical_and(prune_mask, scores_mask)
        self.prune_points(final_prune)

    

    def construct_net(self, train=True):

        self.mlp_cont = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=13,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 16,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        self.mlp_view = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3*self.max_sh_rest,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
    
        self.mlp_dc = tcnn.Network(
            n_input_dims=16,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        
        self.mlp_opacity = tcnn.Network(
            n_input_dims=16,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.mlp_rotation = tcnn.Network(
            n_input_dims=13 + self.rot_feature_dim,
            n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        if train:
            self.net_enabled = True
            # self._features_static = nn.Parameter(self._features_dc[:, 0].clone().detach())
            # self._features_view = nn.Parameter(torch.zeros((self.get_xyz.shape[0], 3), device="cuda").requires_grad_(True))

            if not hasattr(self, '_features_rot') or self._features_rot.numel() == 0:
                with torch.no_grad():
                    rot_norm = torch.nn.functional.normalize(self._rotation, dim=-1)
                    self._features_rot = nn.Parameter(rot_norm[:, :self.rot_feature_dim].contiguous().requires_grad_(True))

            mlp_params = []
            for params in self.mlp_cont.parameters():
                mlp_params.append(params)
            for params in self.mlp_view.parameters():
                mlp_params.append(params)
            for params in self.mlp_dc.parameters():
                mlp_params.append(params)
            for params in self.mlp_opacity.parameters():
                mlp_params.append(params)
            for params in self.mlp_rotation.parameters():
                mlp_params.append(params)

            self.optimizer_net = torch.optim.Adam(mlp_params, lr=0.01, eps=1e-15)
            self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
            [
                torch.optim.lr_scheduler.LinearLR(
                self.optimizer_net, start_factor=0.01, total_iters=100
            ),
                torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer_net,
                milestones=[1_000, 3_500, 6_000],
                gamma=0.33,
            ),
            ]
            )

    def sort_attribute(self, order, xyz_only=False):
        self._xyz = nn.Parameter(self._xyz[order], requires_grad=True)
        if not xyz_only:
            # self._opacity = nn.Parameter(self._opacity[order], requires_grad=True)
            self._scaling = nn.Parameter(self._scaling[order], requires_grad=True)
            self._rotation = nn.Parameter(self._rotation[order], requires_grad=True)
            if hasattr(self, '_features_rot') and self._features_rot.numel() > 0:
                self._features_rot = nn.Parameter(self._features_rot[order], requires_grad=True)
            if self.dynamic_enabled:
                self._velocity = nn.Parameter(self._velocity[order], requires_grad=True)
                self._acceleration = nn.Parameter(self._acceleration[order], requires_grad=True)
                self._time = nn.Parameter(self._time[order], requires_grad=True)
                self._duration = nn.Parameter(self._duration[order], requires_grad=True)
                self._dynamic_logit = nn.Parameter(
                    self._dynamic_logit[order],
                    requires_grad=not self.dynamic_gate_frozen,
                )
                if self._promotion_ema.numel() == order.shape[0]:
                    self._promotion_ema = self._promotion_ema[order]
                if self._forced_dynamic_mask.numel() == order.shape[0]:
                    self._forced_dynamic_mask = self._forced_dynamic_mask[order]
                if self._committed_dynamic_mask.numel() == order.shape[0]:
                    self._committed_dynamic_mask = self._committed_dynamic_mask[order]
            # self._features_dc = nn.Parameter(self._features_dc[order], requires_grad=True)
            # self._features_rest = nn.Parameter(self._features_rest[order], requires_grad=True)
            self._features_static = nn.Parameter(self._features_static[order], requires_grad=True)
            self._features_view = nn.Parameter(self._features_view[order], requires_grad=True)
            # for i in range(len(self.opacity_indices)):
            #     self.opacity_indices[i] = self.opacity_indices[i][order]
            for i in range(len(self.scale_indices)):
                self.scale_indices[i] = self.scale_indices[i][order]
            for i in range(len(self.rotation_indices)):
                self.rotation_indices[i] = self.rotation_indices[i][order]
            for i in range(len(self.appearance_indices)):
                self.appearance_indices[i] = self.appearance_indices[i][order]
            if self.dynamic_enabled and hasattr(self, "dynamic_indices"):
                for i in range(len(self.dynamic_indices)):
                    self.dynamic_indices[i] = self.dynamic_indices[i][order]

        return
    
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x

    def apply_svq(self, args):
        """Quantize every attribute at once, without the staged fine-tuning gap."""
        self.apply_svq_3d(args)
        self.apply_svq_4d(args)

    def apply_svq_3d(self, args):
        """Quantize the static attributes: scale, rotation feature and appearance.

        OMG4 quantizes 3D attributes first and fine-tunes before touching the
        temporal ones, which keeps the static codebooks from having to absorb the
        much larger temporal reconstruction error all at once.
        """
        self.opacity_codes = []
        self.opacity_indices = []
        self.scale_codes = []
        self.scale_indices = []
        self.rotation_codes = []
        self.rotation_indices = []
        self.appearance_codes = []
        self.appearance_indices = []
        # Left empty so _get_dynamic_attributes keeps reading the raw temporal
        # parameters until apply_svq_4d runs.
        self.dynamic_codes = []
        self.dynamic_indices = []

        code_params = []

        # self.kmeans(self._opacity, self.opacity_codes, self.opacity_indices, args.slice_scale, args.cluster_scale, code_params)
        self.kmeans(self._scaling, self.scale_codes, self.scale_indices, args.slice_scale, args.cluster_scale, code_params)
        self.kmeans(self._features_rot, self.rotation_codes, self.rotation_indices, self.rot_feature_dim, args.cluster_rot, code_params)
        self.kmeans(torch.cat([self._features_static, self._features_view],dim=-1), self.appearance_codes, self.appearance_indices, args.slice_app, args.cluster_app, code_params)
        # self.kmeans(self._features_dc[:,0,:], self.appearance_codes, self.appearance_indices, args.slice_scale, args.cluster_app, code_params)
        # self.kmeans(self.get_features.view(len(self._xyz), -1), self.appearance_codes, self.appearance_indices, args.slice_scale, args.cluster_app, code_params)

        self.optimizer_code = torch.optim.Adam(code_params, lr=getattr(args, 'svq_lr', 1e-4), eps=1e-15)
        self.vq_enabled = True

    def apply_svq_4d(self, args):
        """Quantize the temporal attributes, reusing the codebook optimizer."""
        if not self.dynamic_enabled:
            return
        if not self.vq_enabled:
            raise RuntimeError("apply_svq_4d requires apply_svq_3d to have run first")
        if len(self.dynamic_codes) > 0:
            return

        code_params = []
        dynamic_attrs = torch.cat([self._velocity, self._acceleration, self._time, self._duration], dim=-1)
        dynamic_mask = self.get_dynamic_mask()
        if not bool(dynamic_mask.any()):
            return
        if bool(dynamic_mask.all()):
            self.kmeans(dynamic_attrs, self.dynamic_codes, self.dynamic_indices, args.slice_dynamic, args.cluster_dynamic, code_params)
        else:
            self.kmeans_masked(
                dynamic_attrs,
                dynamic_mask,
                self.dynamic_codes,
                self.dynamic_indices,
                args.slice_dynamic,
                args.cluster_dynamic,
                code_params,
            )

        # A new param group rather than a new optimizer, so the Adam state the 3D
        # codebooks built up during their fine-tuning window survives.
        self.optimizer_code.add_param_group({'params': code_params, 'lr': getattr(args, 'svq_lr', 1e-4)})


    @property
    def get_svq_opacity(self):
        opacity = []
        for i in range(len(self.opacity_codes)):
            opacity.append(self.opacity_codes[i][self.opacity_indices[i]])
        return self.opacity_activation(torch.cat(opacity, dim=-1))

    @property
    def get_svq_scale(self):
        scale = []
        for i in range(len(self.scale_codes)):
            scale.append(self.scale_codes[i][self.scale_indices[i]])
        return self.scaling_activation(torch.cat(scale, dim=-1))

    @property
    def get_svq_rotation(self):
        with torch.no_grad():
            cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
            rot_input = torch.cat([cont_feature, self.get_svq_rot_feature], dim=-1)
            rotation = torch.nn.functional.normalize(self.mlp_rotation(rot_input).float(), dim=-1)
        return rotation

    @property
    def get_svq_rot_feature(self):
        rot_feature = []
        for i in range(len(self.rotation_codes)):
            rot_feature.append(self.rotation_codes[i][self.rotation_indices[i]])
        return torch.cat(rot_feature, dim=-1)

    @property
    def get_svq_appearance(self):
        appearance = []
        for i in range(len(self.appearance_codes)):
            appearance.append(self.appearance_codes[i][self.appearance_indices[i]])
        return torch.cat(appearance, dim=-1)
    
    def kmeans(self, param_data, code_list, index_list, svq_len, n_clusters, code_params):
        try:
            import cupy as cp
            from cuml.cluster import KMeans
        except Exception as error:
            raise RuntimeError(
                "SVQ quantization requires compatible CuPy and RAPIDS cuML installations. "
                "Training without quantization does not require RAPIDS; use --skip_quantize."
            ) from error

        assert param_data.shape[1] % svq_len == 0, "invalid sub-vector length"
        # A codebook can never hold more entries than there are Gaussians to
        # cluster, and cuML errors out rather than clamping.
        n_clusters = min(int(n_clusters), param_data.shape[0])
        for i in range(param_data.shape[1]//svq_len):
            input_cp = cp.asarray(param_data[:, i*svq_len:(i+1)*svq_len].detach().cpu())
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=1)
            labels = kmeans.fit_predict(input_cp)
            cluster_centers = kmeans.cluster_centers_

            codebook = torch.nn.Parameter(torch.from_dlpack(cluster_centers)).cuda()
            index = torch.from_dlpack(labels).cuda().long()

            code_list.append(codebook)
            index_list.append(index)
            code_params.append(codebook) 

    def kmeans_masked(self, param_data, active_mask, code_list, index_list,
                      svq_len, n_clusters, code_params):
        """Quantize 4D attributes without spending codes on static rows."""
        try:
            import cupy as cp
            from cuml.cluster import KMeans
        except Exception as error:
            raise RuntimeError(
                "SVQ quantization requires compatible CuPy and RAPIDS cuML installations. "
                "Training without quantization does not require RAPIDS; use --skip_quantize."
            ) from error

        active_mask = active_mask.reshape(-1).bool()
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            return
        if param_data.shape[1] % svq_len != 0:
            raise ValueError("invalid masked-SVQ sub-vector length")
        n_clusters = min(int(n_clusters), active_count)
        for i in range(param_data.shape[1] // svq_len):
            chunk = param_data[active_mask, i * svq_len:(i + 1) * svq_len]
            input_cp = cp.asarray(chunk.detach().cpu())
            kmeans = KMeans(n_clusters=n_clusters, max_iter=1000, n_init=1)
            labels = torch.from_dlpack(kmeans.fit_predict(input_cp)).cuda().long()
            centers = torch.from_dlpack(kmeans.cluster_centers_).cuda()
            # Index zero is the exact static sentinel.  Its temporal values are
            # irrelevant to rendering because the committed hard gate is zero.
            codebook = nn.Parameter(torch.cat((
                torch.zeros((1, svq_len), dtype=centers.dtype, device=centers.device),
                centers,
            ), dim=0))
            indices = torch.zeros(
                param_data.shape[0], dtype=torch.long, device=param_data.device
            )
            indices[active_mask] = labels + 1
            code_list.append(codebook)
            index_list.append(indices)
            code_params.append(codebook)

    def encode(self):
        save_dict = dict()
        xyz_uint16 = float16_to_uint16(self.get_xyz.half())
        sorted_indices = calculate_morton_order(xyz_uint16.int())
        self.sort_attribute(sorted_indices, xyz_only=False)
        xyz_uint16 = float16_to_uint16(self.get_xyz.half())
        save_dict['xyz'] = compress_gpcc(xyz_uint16)



        # save_dict['opacity_code'] = []
        # save_dict['opacity_index'] = []
        # save_dict['opacity_htable'] = []
        # for i in range(len(self.opacity_codes)):
        #     save_dict['opacity_code'].append(self.opacity_codes[i].half().cpu().numpy())
        #     huf_idx, huf_tab = huffman_encode(self.opacity_indices[i].cpu().numpy())
        #     save_dict['opacity_index'].append(huf_idx)
        #     save_dict['opacity_htable'].append(huf_tab)


        save_dict['scale_code'] = []
        save_dict['scale_index'] = []
        save_dict['scale_htable'] = []
        if len(self.scale_codes) != len(self.scale_indices):
            raise ValueError("Scale codebook/index stream count mismatch")
        for i in range(len(self.scale_codes)):
            if self.scale_indices[i].numel() != self._xyz.shape[0]:
                raise ValueError(
                    f"scale_indices[{i}] has {self.scale_indices[i].numel()} "
                    f"rows for {self._xyz.shape[0]} Gaussians"
                )
            save_dict['scale_code'].append(self.scale_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.scale_indices[i].cpu().numpy())
            save_dict['scale_index'].append(huf_idx)
            save_dict['scale_htable'].append(huf_tab)

        save_dict['rotation_code'] = []
        save_dict['rotation_index'] = []
        save_dict['rotation_htable'] = []
        if len(self.rotation_codes) != len(self.rotation_indices):
            raise ValueError("Rotation codebook/index stream count mismatch")
        for i in range(len(self.rotation_codes)):
            if self.rotation_indices[i].numel() != self._xyz.shape[0]:
                raise ValueError(
                    f"rotation_indices[{i}] has "
                    f"{self.rotation_indices[i].numel()} rows for "
                    f"{self._xyz.shape[0]} Gaussians"
                )
            save_dict['rotation_code'].append(self.rotation_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.rotation_indices[i].cpu().numpy())
            save_dict['rotation_index'].append(huf_idx)
            save_dict['rotation_htable'].append(huf_tab)

        save_dict['app_code'] = []
        save_dict['app_index'] = []
        save_dict['app_htable'] = []
        if len(self.appearance_codes) != len(self.appearance_indices):
            raise ValueError("Appearance codebook/index stream count mismatch")
        for i in range(len(self.appearance_codes)):
            if self.appearance_indices[i].numel() != self._xyz.shape[0]:
                raise ValueError(
                    f"appearance_indices[{i}] has "
                    f"{self.appearance_indices[i].numel()} rows for "
                    f"{self._xyz.shape[0]} Gaussians"
                )
            save_dict['app_code'].append(self.appearance_codes[i].half().cpu().numpy())
            huf_idx, huf_tab = huffman_encode(self.appearance_indices[i].cpu().numpy())
            save_dict['app_index'].append(huf_idx)
            save_dict['app_htable'].append(huf_tab)

        save_dict['rot_feature_dim'] = self.rot_feature_dim

        save_dict['MLP_cont'] = self.mlp_cont.params.half().cpu().numpy()
        save_dict['MLP_dc'] = self.mlp_dc.params.half().cpu().numpy()
        save_dict['MLP_sh'] = self.mlp_view.params.half().cpu().numpy()
        save_dict['MLP_opacity'] = self.mlp_opacity.params.half().cpu().numpy()
        save_dict['MLP_rotation'] = self.mlp_rotation.params.half().cpu().numpy()

        save_dict["MLP_offset"] = {
                k: v.detach().cpu().contiguous().numpy()
                for k, v in self.shs_nn.state_dict().items()
                }

        if self.dynamic_enabled:
            save_dict['dynamic_enabled'] = True
            dynamic_mask = self.get_dynamic_mask().detach().cpu().numpy().astype(np.uint8)
            save_dict['dynamic_gate_count'] = int(dynamic_mask.size)
            save_dict['dynamic_gate_bits'] = np.packbits(
                dynamic_mask, bitorder='little'
            )
            save_dict['dynamic_code'] = []
            save_dict['dynamic_index'] = []
            save_dict['dynamic_htable'] = []
            if len(self.dynamic_codes) != len(self.dynamic_indices):
                raise ValueError("Dynamic codebook/index stream count mismatch")
            for i in range(len(self.dynamic_codes)):
                if self.dynamic_indices[i].numel() != self._xyz.shape[0]:
                    raise ValueError(
                        f"dynamic_indices[{i}] has "
                        f"{self.dynamic_indices[i].numel()} rows for "
                        f"{self._xyz.shape[0]} Gaussians"
                    )
                save_dict['dynamic_code'].append(self.dynamic_codes[i].half().cpu().numpy())
                huf_idx, huf_tab = huffman_encode(self.dynamic_indices[i].cpu().numpy())
                save_dict['dynamic_index'].append(huf_idx)
                save_dict['dynamic_htable'].append(huf_tab)
        else:
            save_dict['dynamic_enabled'] = False

        return save_dict

    def decode(self, save_dict, decompress=True):
        self.vq_enabled = False
        self.net_enabled = False
        self.shoffset_enabled = False

        means_strings = save_dict['xyz']
        xyz_uint16 = decompress_gpcc(means_strings).to('cuda')
        sorted_indices = calculate_morton_order(xyz_uint16.int())
        self._xyz = uint16_to_float16(xyz_uint16).float()
        self.sort_attribute(sorted_indices, xyz_only=True)

        scale = []
        rot_feature = []
        appearance = []
        dynamic = []
        opacity = []
        expected_rows = int(self._xyz.shape[0])
        if save_dict.get('dynamic_enabled', False) and 'dynamic_gate_count' in save_dict:
            gate_count = int(save_dict['dynamic_gate_count'])
            if gate_count != expected_rows:
                raise ValueError(
                    f"Compressed dynamic gate has {gate_count} rows; "
                    f"decoded xyz has {expected_rows}"
                )

        if decompress:
            for i in range(len(save_dict['scale_code'])):
                labels = huffman_decode(
                    save_dict['scale_index'][i],
                    save_dict['scale_htable'][i],
                    expected_count=expected_rows,
                    stream_name=f"scale_index[{i}]",
                )
                cluster_centers = save_dict['scale_code'][i]
                scale.append(torch.tensor(cluster_centers[labels]).cuda())
            self._scaling = torch.cat(scale, dim=-1).float()

            for i in range(len(save_dict['rotation_code'])):
                labels = huffman_decode(
                    save_dict['rotation_index'][i],
                    save_dict['rotation_htable'][i],
                    expected_count=expected_rows,
                    stream_name=f"rotation_index[{i}]",
                )
                cluster_centers = save_dict['rotation_code'][i]
                rot_feature.append(torch.tensor(cluster_centers[labels]).cuda())
            self._features_rot = torch.cat(rot_feature, dim=-1).float()

            for i in range(len(save_dict['app_code'])):
                labels = huffman_decode(
                    save_dict['app_index'][i],
                    save_dict['app_htable'][i],
                    expected_count=expected_rows,
                    stream_name=f"app_index[{i}]",
                )
                cluster_centers = save_dict['app_code'][i]
                appearance.append(torch.tensor(cluster_centers[labels]).cuda())
            app_feature = torch.cat(appearance, dim=-1).float()

            if save_dict.get('dynamic_enabled', False) and 'dynamic_code' in save_dict:
                for i in range(len(save_dict['dynamic_code'])):
                    labels = huffman_decode(
                        save_dict['dynamic_index'][i],
                        save_dict['dynamic_htable'][i],
                        expected_count=expected_rows,
                        stream_name=f"dynamic_index[{i}]",
                    )
                    cluster_centers = save_dict['dynamic_code'][i]
                    dynamic.append(torch.tensor(cluster_centers[labels]).cuda())

            self.mlp_cont.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_cont']).cuda().half().requires_grad_(True))
            self.mlp_dc.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_dc']).cuda().half().requires_grad_(True))
            self.mlp_view.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_sh']).cuda().half().requires_grad_(True))
            self.mlp_opacity.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_opacity']).cuda().half().requires_grad_(True))
            self.mlp_rotation.params = torch.nn.Parameter(torch.tensor(save_dict['MLP_rotation']).cuda().half().requires_grad_(True))

        else:
            for i in range(len(self.scale_codes)):
                scale.append(self.scale_codes[i][self.scale_indices[i]])
            self._scaling = torch.cat(scale, dim=-1).float()

            for i in range(len(self.rotation_codes)):
                rot_feature.append(self.rotation_codes[i][self.rotation_indices[i]])
            self._features_rot = torch.cat(rot_feature, dim=-1).float()

            for i in range(len(self.appearance_codes)):
                appearance.append(self.appearance_codes[i][self.appearance_indices[i]])
            app_feature = torch.cat(appearance, dim=-1).float()

            if self.dynamic_enabled and hasattr(self, "dynamic_codes"):
                for i in range(len(self.dynamic_codes)):
                    dynamic.append(self.dynamic_codes[i][self.dynamic_indices[i]])

        decoded_groups = {
            "scale": scale,
            "rotation": rot_feature,
            "appearance": appearance,
            "dynamic": dynamic,
        }
        for group_name, tensors in decoded_groups.items():
            for chunk_index, tensor in enumerate(tensors):
                if tensor.shape[0] != expected_rows:
                    raise ValueError(
                        f"Decoded {group_name}[{chunk_index}] has "
                        f"{tensor.shape[0]} rows; expected {expected_rows}"
                    )

        if save_dict.get('dynamic_enabled', False) and dynamic:
            dynamic_attrs = torch.cat(dynamic, dim=-1).float()
            self._velocity = nn.Parameter(dynamic_attrs[:, 0:3].requires_grad_(True))
            if dynamic_attrs.shape[1] >= 8:
                self._acceleration = nn.Parameter(dynamic_attrs[:, 3:6].requires_grad_(True))
                self._time = nn.Parameter(dynamic_attrs[:, 6:7].requires_grad_(True))
                self._duration = nn.Parameter(dynamic_attrs[:, 7:8].requires_grad_(True))
            else:
                self._acceleration = nn.Parameter(torch.zeros_like(self._velocity).requires_grad_(True))
                self._time = nn.Parameter(dynamic_attrs[:, 3:4].requires_grad_(True))
                self._duration = nn.Parameter(dynamic_attrs[:, 4:5].requires_grad_(True))
            self.dynamic_enabled = True
        elif save_dict.get('dynamic_enabled', False) and 'velocity' in save_dict:
            self._velocity = nn.Parameter(torch.tensor(save_dict['velocity']).cuda().float().requires_grad_(True))
            acceleration = save_dict.get('acceleration', np.zeros_like(save_dict['velocity']))
            self._acceleration = nn.Parameter(torch.tensor(acceleration).cuda().float().requires_grad_(True))
            self._time = nn.Parameter(torch.tensor(save_dict['time']).cuda().float().requires_grad_(True))
            self._duration = nn.Parameter(torch.tensor(save_dict['duration']).cuda().float().requires_grad_(True))
            self.dynamic_enabled = True
        else:
            self._velocity = torch.empty(0)
            self._acceleration = torch.empty(0)
            self._time = torch.empty(0)
            self._duration = torch.empty(0)
            self._dynamic_logit = torch.empty(0)
            self._forced_dynamic_mask = torch.empty(0, dtype=torch.bool)
            self._committed_dynamic_mask = torch.empty(0, dtype=torch.bool)
            self._promotion_ema = torch.empty(0)
            self.dynamic_gate_frozen = False
            self.dynamic_enabled = False

        if self.dynamic_enabled:
            gate_count = int(save_dict.get('dynamic_gate_count', self._xyz.shape[0]))
            if 'dynamic_gate_bits' in save_dict:
                gate_array = np.unpackbits(
                    np.asarray(save_dict['dynamic_gate_bits'], dtype=np.uint8),
                    bitorder='little',
                )[:gate_count]
                if gate_count != self._xyz.shape[0]:
                    raise ValueError(
                        f"Decoded dynamic gate has {gate_count} entries for "
                        f"{self._xyz.shape[0]} Gaussians"
                    )
                committed = torch.tensor(
                    gate_array.astype(np.bool_), dtype=torch.bool, device='cuda'
                )
            else:
                # Backward compatibility for compressed artifacts predating
                # the strict split: every temporal primitive was dynamic.
                committed = torch.ones(
                    self._xyz.shape[0], dtype=torch.bool, device='cuda'
                )
            self._committed_dynamic_mask = committed
            self._forced_dynamic_mask = committed.clone()
            logit_magnitude = self._probability_to_logit(0.999)
            self._dynamic_logit = nn.Parameter(
                torch.where(
                    committed[:, None],
                    torch.full((self._xyz.shape[0], 1), logit_magnitude, device='cuda'),
                    torch.full((self._xyz.shape[0], 1), -logit_magnitude, device='cuda'),
                ),
                requires_grad=False,
            )
            self._promotion_ema = torch.zeros_like(self._dynamic_logit)
            self.dynamic_gate_frozen = True
            self.dynamic_gate_training = False

        cont_feature = self.mlp_cont(self._contract_xyz(self.get_xyz.detach()))
        space_feature = torch.cat([cont_feature, app_feature[:,0:3]],dim=-1)
        view_feature = torch.cat([cont_feature, app_feature[:,3:6]],dim=-1)

        self._features_rest = self.mlp_view(view_feature).reshape(-1,self.max_sh_rest,3).float()
        self._features_dc = self.mlp_dc(space_feature).reshape(-1,1,3).float()
        self._opacity = self.mlp_opacity(space_feature).float()

        rot_input = torch.cat([cont_feature, self._features_rot], dim=-1)
        self._rotation = torch.nn.functional.normalize(self.mlp_rotation(rot_input).float(), dim=-1)

        del self._features_static
        del self._features_view
        del self._features_rot

        mlp_state = {k: torch.from_numpy(v) for k, v in save_dict["MLP_offset"].items()}
        self.shs_nn.load_state_dict(mlp_state)

        sh_offset =  self.get_features_offset(self.get_features, self.get_opacity)

        self._features_dc = self._features_dc + sh_offset[:, 0:1]
        self._features_rest = self._features_rest + sh_offset[:, 1:]

        del self.shs_nn
