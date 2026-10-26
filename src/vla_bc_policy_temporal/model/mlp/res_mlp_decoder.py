from typing import Optional
from warnings import warn

import numpy as np
import torch
from torch import nn

from vla_bc_policy_temporal.model.utility import NormType, ActivateFnType, ActivateFnDict

class ResMlpBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: float = 4,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        hidden_dim = int(dim * expansion)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x: torch.Tensor):
        return x + self.net(x)

class ResMlpDecoder(nn.Module):
    def __init__(
        self,

        num_in_feats: int,
        num_out_feats: int,
        hidden_dim: int,

        num_blocks: int = 2,
        expansion: float = 4,
        dropout_rate: float = 0.1,
        
        is_output_proj: bool = True,
    ) -> None:
        '''__init__ MLP 解码器

        Args:
            num_in_feats (int): 输入特征
            num_out_feats (int): 输出特征
            hidden_dim (int): RES MLP 块的输入特征
            num_blocks (int, optional): RES MLP 块的个数. Defaults to 2.
            expansion (float, optional): RES MLP 块的膨胀系数. Defaults to 4.
            dropout_rate (float, optional): RES MLP 块的 dropout. Defaults to 0.1.

            is_output_proj (bool, optional): 输出层是否有额外 Linear 用于将 hidden feat 调整为 output feat，当编码器输出即结果时可设为 True，取 False 时 hidden feat 要等于 output feat. Defaults to True.
        '''
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(num_in_feats, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        if is_output_proj:
            self.output_proj = nn.Linear(hidden_dim, num_out_feats)
        else:
            self.output_proj = nn.Identity()
            if hidden_dim != num_out_feats:
                raise ValueError(
                    "ResMlpDecoder with is_output_proj=False requires "
                    f"hidden_dim == num_out_feats (got {hidden_dim} != {num_out_feats})"
                )

        self.blocks = nn.Sequential(
            *[
                ResMlpBlock(
                    dim = hidden_dim,
                    expansion = expansion,
                    dropout = dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor):
        h = self.input_proj(x)
        h = self.blocks(h)
        y = self.output_proj(h)
        return y