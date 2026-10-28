"""Independent MS-HAB extractor, without edits to existing model modules."""
from __future__ import annotations

from typing import Mapping

import torch

from vla_bc_policy_temporal.dataset.utility import IMAGE_OBS_KEY, VECTION_OBS_KEY
from vla_bc_policy_temporal.model.extractor import CurImgTempVecTrajExtractor


class RepViTHistorySingleStepExtractor(CurImgTempVecTrajExtractor):
    """Two current depth views, 5 state tokens, 9 reference tokens -> 512D.

    Assumes the plugin registry keys are registered by
    policy.repvit_history_single_step_policy before construction.
    """

    def __init__(
        self,
        *args,
        vec_temproal_encoder_type: str = "repvit_history_residual",
        traj_temproal_encoder_type: str = "repvit_masked_dw_ref",
        **kwargs,
    ) -> None:
        if vec_temproal_encoder_type != "repvit_history_residual":
            raise ValueError("This extractor requires repvit_history_residual")
        if traj_temproal_encoder_type != "repvit_masked_dw_ref":
            raise ValueError("This extractor requires repvit_masked_dw_ref")
        super().__init__(
            *args,
            vec_temproal_encoder_type=vec_temproal_encoder_type,
            traj_temproal_encoder_type=traj_temproal_encoder_type,
            **kwargs,
        )
        if not self.use_vector or not self.use_traj:
            raise ValueError("History and reference branches are required")

    def forward(
        self,
        obs: Mapping[str, Mapping[str, torch.Tensor]],
        traject: torch.Tensor | None = None,
        masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        masks = {} if masks is None else masks
        features: dict[str, torch.Tensor] = {}
        images, vectors = obs[IMAGE_OBS_KEY], obs[VECTION_OBS_KEY]
        obs_valid = masks.get("obs_valid_mask")
        traj_valid = masks.get("traject_valid_mask")

        if obs_valid is not None:
            if obs_valid.ndim != 2 or obs_valid.shape[1] != self.vec_horizon:
                raise ValueError("obs_valid_mask must be [B,5]")
            if not bool(obs_valid[:, -1].bool().all()):
                raise ValueError("Current observation must be valid")
        if traj_valid is not None:
            if traj_valid.ndim != 2 or traj_valid.shape[1] != self.traj_seq_len:
                raise ValueError("traject_valid_mask must be [B,9]")

        for i, cam in enumerate(self.img_use_key):
            depth = images[cam]
            if depth.ndim != 5 or depth.shape[2] != self.img_obs_dim[cam]:
                raise ValueError(f"Camera {cam} expected [B,T,C,H,W], got {tuple(depth.shape)}")
            visual = self.image_encoders[str(i)](depth[:, -1].contiguous())
            if visual.ndim != 2 or visual.shape[1] != self.image_feat_dim[cam]:
                raise RuntimeError(f"Camera {cam} output has incorrect feature dimension")
            features[f"image:{cam}"] = visual

        states = [vectors[name] for name in self.vec_use_key]
        batch = states[0].shape[0]
        for name, value in zip(self.vec_use_key, states):
            if value.shape != (batch, self.vec_horizon, self.vec_obs_dim[name]):
                raise ValueError(f"State {name}: incorrect shape {tuple(value.shape)}")
        state = self.vec_feat_encoder(torch.cat(states, dim=-1))
        features["vector"] = self.vec_temproal_encoder(state, valid_mask=obs_valid)

        if traject is None or traject.ndim != 3 or traject.shape[1:] != (
            self.traj_seq_len, self.traj_dim
        ):
            raise ValueError(f"Reference must be [B,{self.traj_seq_len},{self.traj_dim}]")
        ref = self.traj_feat_encoder(traject)
        features["traject"] = self.traj_temproal_encoder(ref, valid_mask=traj_valid)
        return self.fusion(features)
