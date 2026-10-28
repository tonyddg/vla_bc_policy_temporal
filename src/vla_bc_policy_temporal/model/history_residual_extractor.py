"""Opt-in feature extractor with an explicit masked state-history branch.

The original CurImgTempVecTrajExtractor is inherited, not replaced.  Only
its forward path differs: obs_valid_mask is delivered to HistoryResidualMLP.
The reference CNN, image encoders, and fusion logic keep their old behavior.
"""

from typing import Mapping

import torch

from vla_bc_policy_temporal.dataset.utility import IMAGE_OBS_KEY, VECTION_OBS_KEY
from vla_bc_policy_temporal.model.extractor import CurImgTempVecTrajExtractor
from vla_bc_policy_temporal.model.history_residual_encoder import HistoryResidualMLP


class HistoryResidualCurImgVecTrajExtractor(CurImgTempVecTrajExtractor):
    """History-aware variant; keeps all existing feature dimensions."""

    def __init__(self, *args, vec_temproal_encoder_type="history_residual_mlp", **kwargs):
        if vec_temproal_encoder_type != "history_residual_mlp":
            raise ValueError(
                "HistoryResidualCurImgVecTrajExtractor requires "
                "vec_temproal_encoder_type='history_residual_mlp'"
            )
        super().__init__(
            *args,
            vec_temproal_encoder_type=vec_temproal_encoder_type,
            **kwargs,
        )
        if not self.use_vector or not isinstance(self.vec_temproal_encoder, HistoryResidualMLP):
            raise ValueError("The history-residual extractor requires vector observations")

    def forward(
        self,
        obs: Mapping[str, Mapping[str, torch.Tensor]],
        traject: torch.Tensor | None = None,
        masks: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        masks = masks or {}
        feature: dict[str, torch.Tensor] = {}
        img_obs = obs[IMAGE_OBS_KEY]
        vec_obs = obs[VECTION_OBS_KEY]
        obs_mask = masks.get("obs_valid_mask")
        traj_mask = masks.get("traject_valid_mask")
        if obs_mask is not None and (
            obs_mask.ndim != 2 or obs_mask.shape[1] != self.vec_horizon
        ):
            raise ValueError("obs_valid_mask must have shape [B, observation_horizon]")
        if obs_mask is not None and not obs_mask[:, -1].bool().all():
            raise ValueError("Current image/observation must be valid")

        # Unchanged: use only the most recent image from each selected camera.
        for i, cam in enumerate(self.img_use_key):
            x = img_obs[cam]
            if x.ndim != 5 or x.shape[2] != self.img_obs_dim[cam]:
                raise ValueError(
                    f"Image {cam} must be [B,T,{self.img_obs_dim[cam]},H,W]; "
                    f"got {tuple(x.shape)}"
                )
            y = self.image_encoders[str(i)](x[:, -1].contiguous())
            if y.ndim != 2 or y.shape[1] != self.image_feat_dim[cam]:
                raise ValueError(
                    f"Backbone for {cam} returned {tuple(y.shape)}, "
                    f"configured {self.image_feat_dim[cam]} features"
                )
            feature[f"image:{cam}"] = y

        if self.use_vector:
            vectors = [vec_obs[name] for name in self.vec_use_key]
            batch = vectors[0].shape[0]
            for name, x in zip(self.vec_use_key, vectors):
                expected = (batch, self.vec_horizon, self.vec_obs_dim[name])
                if x.shape != expected:
                    raise ValueError(
                        f"Vector {name} got {tuple(x.shape)}, expected {expected}"
                    )
            x = torch.cat(vectors, dim=-1)
            if obs_mask is not None and self.is_set_padding_to_zero:
                x = x * obs_mask.to(x.device, dtype=x.dtype).unsqueeze(-1)
            x = self.vec_feat_encoder(x)
            if obs_mask is not None and self.is_set_padding_to_zero:
                x = x * obs_mask.to(x.device, dtype=x.dtype).unsqueeze(-1)
            # Only new behavior: pass the observation mask to the history MLP.
            feature["vector"] = self.vec_temproal_encoder(x, valid_mask=obs_mask)

        # Unchanged: reference trajectory CNN and its existing padding option.
        if self.use_traj:
            if traject is None or traject.ndim != 3 or traject.shape[1:] != (
                self.traj_seq_len, self.traj_dim
            ):
                raise ValueError(
                    f"Trajectory must be [B,{self.traj_seq_len},{self.traj_dim}]"
                )
            t = traject
            if traj_mask is not None and self.is_set_padding_to_zero:
                t = t * traj_mask.to(t.device, dtype=t.dtype).unsqueeze(-1)
            t = self.traj_feat_encoder(t)
            if traj_mask is not None and self.is_set_padding_to_zero:
                t = t * traj_mask.to(t.device, dtype=t.dtype).unsqueeze(-1)
            feature["traject"] = self.traj_temproal_encoder(t)

        return self.fusion(feature)
