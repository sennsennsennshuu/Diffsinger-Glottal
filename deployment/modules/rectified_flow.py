from __future__ import annotations

from typing import List, Tuple

import torch

from modules.core import (
    RectifiedFlow, PitchRectifiedFlow, MultiVarianceRectifiedFlow
)
from modules.backbone import BACKBONES
from utils.hparams import hparams


class RectifiedFlowONNX(RectifiedFlow):
    # noinspection PyMissingConstructor
    def __init__(self, out_dims, num_feats, t_start, time_scale_factor,
                 backbone_type, backbone_args, spec_min, spec_max):
        """Bridge v2 ONNX init args → v3 RectifiedFlow.__init__.

        The original v2 ONNX code passed v2-style keyword args.  We translate
        them into the v3 format (sample_dim + backbone Module).
        """
        sample_dim = out_dims * num_feats
        cls = BACKBONES[backbone_type]
        # Map v2-style backbone_args keys to v3 LYNXNet/WaveNet keys
        bb = {k.replace('dropout_rate', 'dropout'): v for k, v in backbone_args.items()}
        backbone = cls(sample_dim, hparams['hidden_size'], **bb)
        super().__init__(
            sample_dim=sample_dim,
            backbone=backbone,
            t_start=t_start,
            time_scale_factor=time_scale_factor,
        )
        self.out_dims = out_dims
        self.num_feats = num_feats
        # Convert to buffer — follows model device (CUDA trace needs GPU tensors)
        self.register_buffer('spec_min', torch.as_tensor(spec_min, dtype=torch.float32))
        self.register_buffer('spec_max', torch.as_tensor(spec_max, dtype=torch.float32))
    @property
    def backbone(self):
        return self.velocity_fn

    # We give up the setter for the property `backbone` because this will cause TorchScript to fail
    # @backbone.setter
    @torch.jit.unused
    def set_backbone(self, value):
        self.velocity_fn = value

    def sample_euler(self, x, t, dt: float, cond):
        x += self.velocity_fn(x, t * self.time_scale_factor, cond) * dt
        return x

    def norm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return (x - b) / k

    def denorm_spec(self, x):
        k = (self.spec_max - self.spec_min) / 2.
        b = (self.spec_max + self.spec_min) / 2.
        return x * k + b

    def forward(self, condition, x_end=None, depth=None, steps=10):
        condition = condition.transpose(1, 2)  # [1, T, H] => [1, H, T]
        device = condition.device
        n_frames = condition.shape[2]
        noise = torch.randn((1, self.num_feats, self.out_dims, n_frames), device=device)
        if x_end is None:
            t_start = 0.
            x = noise
        else:
            t_start = torch.max(1 - depth, torch.tensor(self.t_start, dtype=torch.float32, device=device))
            # x_end from aux_decoder is already in [-1, 1] (same range as the
            # RectifiedFlow's internal representation). Training RectifiedFlow
            # has no normalizer — the DiffusionDecoder handles it externally.
            # ONNX RectifiedFlow.norm_spec would double-normalize, corrupting
            # the aux output.  Skip norm for shallow-diffusion x_end.
            x_end = x_end.transpose(-2, -1)
            if self.num_feats == 1:
                x_end = x_end[:, None, :, :]
            if t_start <= 0.:
                x = noise
            elif t_start >= 1.:
                x = x_end
            else:
                x = t_start * x_end + (1 - t_start) * noise

        t_width = 1. - t_start
        if t_width >= 0.:
            dt = t_width / max(1, steps)
            for t in torch.arange(steps, dtype=torch.long, device=device)[:, None].float() * dt + t_start:
                x = self.sample_euler(x, t, dt, condition)

        if self.num_feats == 1:
            x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, M, T] => [B, T, M]
        else:
            x = x.permute(0, 1, 3, 2)  # [B, F, M, T] => [B, F, T, M]
        x = self.denorm_spec(x)
        return x


class PitchRectifiedFlowONNX(RectifiedFlowONNX, PitchRectifiedFlow):
    def __init__(self, vmin: float, vmax: float,
                 cmin: float, cmax: float, repeat_bins,
                 time_scale_factor=1000,
                 backbone_type=None, backbone_args=None):
        self.vmin = vmin
        self.vmax = vmax
        self.cmin = cmin
        self.cmax = cmax
        # Build backbone from v2-style args (same pattern as RectifiedFlowONNX)
        bb_type = backbone_type or hparams.get('backbone_type', 'lynxnet')
        bb = {k.replace('dropout_rate', 'dropout'): v
              for k, v in (backbone_args or {}).items()}
        cls = BACKBONES[bb_type]
        backbone = cls(repeat_bins, hparams['hidden_size'], **bb)
        # Call RectifiedFlow.__init__ directly.
        # super() chain is broken because PitchRectifiedFlow (=_RectifiedFlowDummy)
        # sits between RectifiedFlowONNX and RectifiedFlow in MRO, but
        # RectifiedFlowONNX uses # noinspection PyMissingConstructor and
        # doesn't call super().__init__() at all.
        RectifiedFlow.__init__(
            self,
            sample_dim=repeat_bins,
            backbone=backbone,
            t_start=0.,
            time_scale_factor=time_scale_factor,
        )
        self.register_buffer('spec_min', torch.tensor([vmin], dtype=torch.float32))
        self.register_buffer('spec_max', torch.tensor([vmax], dtype=torch.float32))
        self.num_feats = 1
        self.out_dims = repeat_bins

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x

    def clamp_spec(self, x):
        return x.clamp(min=self.cmin, max=self.cmax)

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2.
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m
        x = x.mean(dim=-1)
        return x


class MultiVarianceRectifiedFlowONNX(RectifiedFlowONNX, MultiVarianceRectifiedFlow):
    def __init__(
            self, ranges: List[Tuple[float, float]],
            clamps: List[Tuple[float | None, float | None] | None],
            repeat_bins, time_scale_factor=1000,
            backbone_type=None, backbone_args=None
    ):
        assert len(ranges) == len(clamps)
        self.clamps = clamps
        vmin = [r[0] for r in ranges]
        vmax = [r[1] for r in ranges]
        # Build backbone from v2-style args
        bb_type = backbone_type or hparams.get('backbone_type', 'lynxnet')
        bb = {k.replace('dropout_rate', 'dropout'): v
              for k, v in (backbone_args or {}).items()}
        cls = BACKBONES[bb_type]
        backbone = cls(repeat_bins, hparams['hidden_size'], **bb)
        RectifiedFlow.__init__(
            self,
            sample_dim=repeat_bins,
            backbone=backbone,
            t_start=0.,
            time_scale_factor=time_scale_factor,
        )
        # spec_min/max are repeated per-variance-bin so denorm_spec broadcast works.
        # e.g. [breathiness_min, voicing_min, tension_min] × 16 bins each → shape [48].
        num_vars = len(vmin)
        bins_per_var = repeat_bins // num_vars
        spec_min_vals = [torch.full((bins_per_var,), v, dtype=torch.float32) for v in vmin]
        spec_max_vals = [torch.full((bins_per_var,), v, dtype=torch.float32) for v in vmax]
        self.register_buffer('spec_min', torch.cat(spec_min_vals))
        self.register_buffer('spec_max', torch.cat(spec_max_vals))
        self.num_feats = 1  # single combined feature channel
        self.out_dims = repeat_bins

    def clamp_spec(self, x):
        if isinstance(x, list):
            return [xi.clamp(min=c[0], max=c[1]) if c[0] is not None and c[1] is not None else xi
                    for xi, c in zip(x, self.clamps)]
        # Single tensor: use first clamp pair
        c = self.clamps[0] if self.clamps else (None, None)
        if c[0] is not None and c[1] is not None:
            return x.clamp(min=c[0], max=c[1])
        return x

    def denorm_spec(self, x):
        # x: [B, T, total_repeat_bins] (squeeze+permuted in RectifiedFlowONNX.forward)
        d = (self.spec_max - self.spec_min) / 2.  # [total_repeat_bins]
        m = (self.spec_max + self.spec_min) / 2.
        x = x * d + m  # [B, T, total_repeat_bins]
        # Reshape to per-variance: [B, T, num_vars, bins_per_var] → mean → [B, num_vars, T]
        B, T_size = x.shape[0], x.shape[1]
        num_vars = len(self.clamps)
        bins_per_var = self.out_dims // num_vars
        x = x.view(B, T_size, num_vars, bins_per_var).mean(dim=-1)  # [B, T, num_vars]
        x = x.permute(0, 2, 1)  # [B, num_vars, T]
        return x
