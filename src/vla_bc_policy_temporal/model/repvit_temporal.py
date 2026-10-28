"""Opt-in temporal encoders for the MS-HAB RepViT single-step policy.

Keeps the original time horizons: 5 states and 9 references (4 past,
current, 4 future). Does not replace the project's existing GRU or CNN.
"""
from __future__ import annotations

import torch
from torch import nn


class RepViTHistoryResidualMLP(nn.Module):
    """Lightweight state-history encoder with current-state skip connection.

    Input [B,5,128], output [B,128]. Masked historical positions do not
    create fake motion at episode boundaries. Current state is the last token.
    """

    def __init__(
        self,
        input_dim: int = 128,
        out_dim: int = 128,
        seq_len: int = 5,
        hidden_dim: int = 128,
        extract_idx: int = -1,
        initial_history_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len < 2 or extract_idx != -1:
            raise ValueError("Requires seq_len>=2 and current state at last index")
        if min(input_dim, out_dim, hidden_dim) <= 0:
            raise ValueError("All feature dimensions must be positive")
        self.seq_len, self.input_dim = int(seq_len), int(input_dim)
        self.current_proj = (nn.Identity() if input_dim == out_dim
                             else nn.Linear(input_dim, out_dim))
        self.history_encoder = nn.Sequential(
            nn.Linear((seq_len - 1) * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.history_scale = nn.Parameter(torch.tensor(float(initial_history_scale)))

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.input_dim):
            raise ValueError(f"State history must be [B,{self.seq_len},{self.input_dim}]")
        current = x[:, -1]
        diff = x[:, :-1] - current.unsqueeze(1)
        has_history = None
        if valid_mask is not None:
            if tuple(valid_mask.shape) != tuple(x.shape[:2]):
                raise ValueError("State valid_mask must match [B,T]")
            mask = valid_mask.to(device=x.device, dtype=torch.bool)
            if not bool(mask[:, -1].all()):
                raise ValueError("The current state must be valid")
            diff = diff.masked_fill(~mask[:, :-1, None], 0.0)
            has_history = mask[:, :-1].any(dim=1, keepdim=True)
        history = self.history_encoder(diff.flatten(1))
        if has_history is not None:
            history = history * has_history.to(history.dtype)
        return self.current_proj(current) + self.history_scale * history


class _DWTemporalBlock(nn.Module):
    """Depthwise temporal convolution followed by pointwise channel MLP."""

    def __init__(self, channels: int, kernel_size: int = 5, expansion: int = 2):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 != 1:
            raise ValueError("Temporal kernel_size must be a positive odd integer")
        self.norm = nn.LayerNorm(channels)
        self.dw = nn.Conv1d(
            channels, channels, kernel_size, padding=kernel_size // 2,
            groups=channels, bias=False,
        )
        self.pw = nn.Sequential(
            nn.Conv1d(channels, channels * expansion, 1),
            nn.GELU(),
            nn.Conv1d(channels * expansion, channels, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,C,T], mask: [B,1,T]. Apply mask AFTER each block too:
        # invalid tokens cannot contaminate the following temporal block.
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        z = self.pw(self.dw(z * mask))
        return (x + z) * mask


class RepViTMaskedDWReferenceEncoder(nn.Module):
    """9-point reference encoder, [B,9,128] -> [B,128].

    Preserves the original direct current-reference skip connection:
      reference_feature = Linear([masked_mean(TCN(z)), z_current]).
    """

    def __init__(
        self,
        input_dim: int = 128,
        out_dim: int = 128,
        seq_len: int = 9,
        hidden_dim: int = 128,
        extract_idx: int = 4,
        kernel_size: int = 5,
        num_blocks: int = 2,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        if min(input_dim, out_dim, hidden_dim, seq_len, num_blocks, expansion) <= 0:
            raise ValueError("Encoder dimensions must be positive")
        if not (0 <= extract_idx < seq_len):
            raise ValueError("extract_idx must select current reference token")
        self.seq_len, self.input_dim, self.extract_idx = seq_len, input_dim, extract_idx
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [_DWTemporalBlock(hidden_dim, kernel_size, expansion) for _ in range(num_blocks)]
        )
        self.out_proj = nn.Linear(2 * hidden_dim, out_dim)

    def forward(
        self, x: torch.Tensor, valid_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.input_dim):
            raise ValueError(f"Reference must be [B,{self.seq_len},{self.input_dim}]")
        if valid_mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            if tuple(valid_mask.shape) != tuple(x.shape[:2]):
                raise ValueError("Reference valid_mask must match [B,T]")
            mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if not bool(mask[:, self.extract_idx].all()):
            raise ValueError("The current reference point must be valid")

        encoded = self.input_proj(x)
        current = encoded[:, self.extract_idx]
        weights = mask.unsqueeze(1).to(encoded.dtype)
        y = encoded.transpose(1, 2) * weights
        for block in self.blocks:
            y = block(y, weights)
        global_feature = (y * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)
        return self.out_proj(torch.cat([global_feature, current], dim=-1))
