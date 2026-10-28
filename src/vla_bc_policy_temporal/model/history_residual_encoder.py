"""Opt-in lightweight state-history encoder.

This module does not modify the existing GRUTemproalEncoder, or any of the
other temporal encoders.  Select it via HistoryResidualBCPolicy only.
"""

import torch
from torch import nn


class HistoryResidualMLP(nn.Module):
    """Combine the current state embedding with a short history residual.

    Input:  [batch, history_len, embedding_dim], current state is the last step.
    Output: [batch, out_dim].

    Only valid historical positions contribute.  Padded positions are masked
    **after** feature-difference computation, so boundary padding cannot
    introduce an artificial motion signal.  The history scale is trainable.
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
        if seq_len < 2:
            raise ValueError("HistoryResidualMLP requires at least 2 states")
        if extract_idx != -1:
            raise ValueError("HistoryResidualMLP expects the last state as current")
        if min(input_dim, out_dim, hidden_dim) <= 0:
            raise ValueError("All feature dimensions must be positive")
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.current_proj = (
            nn.Identity() if input_dim == out_dim else nn.Linear(input_dim, out_dim)
        )
        self.history_encoder = nn.Sequential(
            nn.Linear((seq_len - 1) * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.history_scale = nn.Parameter(
            torch.tensor(float(initial_history_scale))
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.input_dim):
            raise ValueError(
                f"Expected [B,{self.seq_len},{self.input_dim}], got {tuple(x.shape)}"
            )

        current = x[:, -1]
        diff = x[:, :-1] - current.unsqueeze(1)
        has_history = None
        if valid_mask is not None:
            if valid_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Expected valid_mask {tuple(x.shape[:2])}, got {tuple(valid_mask.shape)}"
                )
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
            if not valid_mask[:, -1].all():
                raise ValueError("The current observation must be valid")
            diff = diff.masked_fill(~valid_mask[:, :-1].unsqueeze(-1), 0.0)
            has_history = valid_mask[:, :-1].any(dim=1, keepdim=True)

        history = self.history_encoder(diff.flatten(start_dim=1))
        if has_history is not None:
            # A history of only padding must not contribute the MLP's bias.
            history = history * has_history.to(dtype=history.dtype)
        return self.current_proj(current) + self.history_scale * history
