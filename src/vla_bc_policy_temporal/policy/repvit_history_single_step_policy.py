"""Opt-in policy: RepViT-M0.9 + history residual + masked DW reference.

No existing classes or registry entries are overwritten. Lightning forwards
[B,1,10] for compatibility with existing BCPolicy loss and normalization.
For physical single-step execution use predict_single_action(...): [B,10].
"""
from __future__ import annotations

from typing import Mapping

import torch

from vla_bc_policy_temporal.encoder import ImageEncoderDict
from vla_bc_policy_temporal.encoder.repvit_multiscale import RepViTMultiScaleDepthEncoder
from vla_bc_policy_temporal.model.temproal_encoder import TemproalEncoderDict
from vla_bc_policy_temporal.model.repvit_temporal import (
    RepViTHistoryResidualMLP,
    RepViTMaskedDWReferenceEncoder,
)
from vla_bc_policy_temporal.model.extractor import ExtractorDict
from vla_bc_policy_temporal.model.repvit_history_extractor import RepViTHistorySingleStepExtractor
from vla_bc_policy_temporal.policy.bc_policy import BCPolicy


IMAGE_KEY = "repvit_multiscale_depth"
STATE_KEY = "repvit_history_residual"
REFERENCE_KEY = "repvit_masked_dw_ref"
EXTRACTOR_KEY = "repvit_history_single_step"

# Only add new, opt-in entries. Original ResNet / CNN / BCPolicy continue to work.
for registry, key, cls in (
    (ImageEncoderDict, IMAGE_KEY, RepViTMultiScaleDepthEncoder),
    (TemproalEncoderDict, STATE_KEY, RepViTHistoryResidualMLP),
    (TemproalEncoderDict, REFERENCE_KEY, RepViTMaskedDWReferenceEncoder),
    (ExtractorDict, EXTRACTOR_KEY, RepViTHistorySingleStepExtractor),
):
    if key in registry and registry[key] is not cls:
        raise RuntimeError(f"Registry key collision: {key}")
    registry[key] = cls


class RepViTHistorySingleStepPolicy(BCPolicy):
    """Single-action MS-HAB policy using existing normalized chunk-shaped API."""

    def __init__(self, extractor_type: str = EXTRACTOR_KEY, **kwargs):
        if extractor_type != EXTRACTOR_KEY:
            raise ValueError(f"Use extractor_type={EXTRACTOR_KEY!r}")
        super().__init__(extractor_type=extractor_type, **kwargs)
        if self.action_horizon != 1:
            raise ValueError("Single-step policy requires action_horizon=1")

    @torch.no_grad()
    def predict_single_action(
        self,
        obs: Mapping[str, Mapping[str, torch.Tensor]],
        traject: torch.Tensor | None = None,
        masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Denormalized current whole-body action, shape [B,10].

        Call policy.eval() before online prediction. This does not send commands
        to the robot; it only predicts the action for the next control tick.
        """
        actions = self.predict_action(obs, traject, masks)
        if actions.ndim != 3 or actions.shape[1] != 1:
            raise RuntimeError("Expected a single-step prediction [B,1,D]")
        return actions[:, 0, :]

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Restoring a fully trained checkpoint must not fetch ImageNet weights.

        Lightning restores all actual backbone weights from `state_dict`.
        Set `pretrained=False` in the saved constructor configuration, while
        keeping the training-time constructor allowed to preload ImageNet.
        """
        super().on_save_checkpoint(checkpoint)
        hparams = checkpoint.get("hyper_parameters", {})
        backbone_cfg = hparams.get("extractor_kwargs", {}).get("img_backbone_cfg", {})
        for cam, cfg in list(backbone_cfg.items()):
            if len(cfg) != 3:
                continue
            encoder_type, encoder_kwargs, output_dim = cfg
            if encoder_type == IMAGE_KEY:
                new_kwargs = dict(encoder_kwargs)
                new_kwargs["pretrained"] = False
                backbone_cfg[cam] = (encoder_type, new_kwargs, output_dim)
