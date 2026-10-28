"""RepViT-M0.9 multi-scale depth encoder (new, opt-in).

Backbone implementation comes from the maintained `timm` RepViT model,
not a hand-written copy. Reference: https://github.com/THU-MIG/RepViT
and https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/repvit.py

Dependency (install in the target training environment):
    pip install 'timm>=1.0.15'

The official ImageNet weights are optional. For pretrained=True, timm
adapts the first RGB convolution to a single depth channel using its
input-channel adaptation path; depth is NOT normalized as ImageNet RGB.
Offline training: pretrained=False (default).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RepViTMultiScaleDepthEncoder(nn.Module):
    """[B,C,H,W] -> [B,out_dim], with three spatial scales (8/16/32).

    Each scale: 1x1 projection -> GELU -> adaptive 2x2 pooling.
    All three scales are concatenated (no conditional attention/gating).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_dim: int = 128,
        pretrained: bool = False,
        model_name: str = "repvit_m0_9.dist_300e_in1k",
        scale_dim: int = 32,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        if min(in_channels, out_dim, scale_dim, pool_size) <= 0:
            raise ValueError("All channel widths and pool_size must be positive")
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "RepViT requires the official timm implementation. "
                "Install in your project environment: pip install 'timm>=1.0.15'"
            ) from exc
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                in_chans=in_channels,
                out_indices=(1, 2, 3),  # stride 8, 16, 32
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot initialize timm RepViT {model_name!r}. "
                "Update timm, and if pretrained=True check network/cache availability."
            ) from exc

        channels = list(self.backbone.feature_info.channels())
        strides = list(self.backbone.feature_info.reduction())
        if len(channels) != 3 or strides != [8, 16, 32]:
            raise RuntimeError(
                "RepViT feature layout changed: expected strides [8,16,32], "
                f"got {strides}, channels={channels}"
            )
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.scale_projs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ch, scale_dim, 1, bias=False), nn.GELU())
            for ch in channels
        )
        self.pool_size = int(pool_size)
        flat_dim = 3 * scale_dim * pool_size * pool_size
        self.out_proj = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected depth images [B,{self.in_channels},H,W], got {tuple(x.shape)}"
            )
        feats = self.backbone(x)
        if len(feats) != 3:
            raise RuntimeError(f"RepViT returned {len(feats)} scales instead of 3")
        pooled = [
            F.adaptive_avg_pool2d(proj(feat), (self.pool_size, self.pool_size))
            for proj, feat in zip(self.scale_projs, feats)
        ]
        return self.out_proj(torch.cat(pooled, dim=1))
