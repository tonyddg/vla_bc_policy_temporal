from typing import Any, Dict, Literal, Mapping, Optional
import torch
from torch import nn

from vla_bc_policy_temporal.model.mlp.mlp_decoder import MlpDecoder

class ActionChunkHead(nn.Module):
    """Single regression head predicting the entire action chunk."""
    def __init__(self, num_in_feats: int, num_out_feats: int, action_horizon: int,
                 decoder_kwargs: Mapping[str, Any] | None = None):
        super().__init__()
        if min(num_in_feats, num_out_feats, action_horizon) <= 0:
            raise ValueError("Action head dimensions must be positive")
        self.action_horizon, self.action_dim = action_horizon, num_out_feats
        cfg = dict(decoder_kwargs or {})
        cfg.pop("num_in_feats", None)
        cfg.pop("num_out_feats", None)
        self.projection = MlpDecoder(
            num_in_feats = num_in_feats,
            num_out_feats = action_horizon * num_out_feats, 
            **cfg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Action head input must be [B,F], got {tuple(x.shape)}")
        return self.projection(x).reshape(x.shape[0], self.action_horizon, self.action_dim)


class MultiActionChunkHead(nn.Module):
    """Independent regression heads for non-overlapping action dimension groups.

    head_decoder_config: {"joint": ((0, 7), {"mlp_layers": [128]}), ...}
    Intervals must cover [0, num_out_feats) exactly: gaps would leave an
    otherwise unconstrained action component in the output.
    """
    def __init__(
            self, 
            num_in_feats: int, 
            num_out_feats: int, 
            action_horizon: int,
            head_decoder_config: Mapping[str, tuple[tuple[int, int], Mapping[str, Any]]]
        ):
        super().__init__()
        if min(num_in_feats, num_out_feats, action_horizon) <= 0:
            raise ValueError("Action head dimensions must be positive")
        if not head_decoder_config:
            raise ValueError("At least one action head group is required")
        self.action_horizon, self.action_dim = action_horizon, num_out_feats
        self.group_ranges: dict[str, tuple[int, int]] = {}
        self.heads = nn.ModuleDict()
        seen: set[int] = set()
        for i, (name, (span, kwargs)) in enumerate(head_decoder_config.items()):
            start, end = map(int, span)
            if not 0 <= start < end <= num_out_feats:
                raise ValueError(f"Invalid output range for {name}: {span}")
            positions = set(range(start, end))
            if seen & positions:
                raise ValueError(f"Overlapping action range for {name}: {span}")
            seen.update(positions)
            self.group_ranges[name] = (start, end)
            cfg = dict(kwargs)
            cfg.pop("num_in_feats", None)
            cfg.pop("num_out_feats", None)
            self.heads[str(i)] = MlpDecoder(
                num_in_feats = num_in_feats,
                num_out_feats = action_horizon * (end - start), **cfg
            )
        if seen != set(range(num_out_feats)):
            raise ValueError(f"Head ranges do not cover action dimensions: {set(range(num_out_feats))-seen}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Action head input must be [B,F], got {tuple(x.shape)}")
        pred = x.new_zeros(x.shape[0], self.action_horizon, self.action_dim)
        for i, (start, end) in enumerate(self.group_ranges.values()):
            pred[:, :, start:end] = self.heads[str(i)](x).reshape(x.shape[0], self.action_horizon, end-start)
        return pred


# New shape-aware multi-head implementation; old single-step model is retired.
MultiActionHead = MultiActionChunkHead
SingleActionHead = ActionChunkHead

ActionHeadType = Literal["single", "multi"]
ActionHeadDict = {"single": ActionChunkHead, "multi": MultiActionChunkHead}

def get_mlp(encoder: ActionHeadType):
    res = ActionHeadDict.get(encoder, None)
    if res is None:
        raise ValueError(f"Unknown action head type {encoder}, available in {list(ActionHeadDict.keys())}")
    return ActionHeadDict[encoder]
