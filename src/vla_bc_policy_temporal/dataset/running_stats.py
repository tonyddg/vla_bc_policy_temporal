"""Streaming statistics for the temporal LeRobot dataset.

One Dataset index corresponds to one real time step.  Statistics therefore
use the *current* observation (last frame), current action (first frame) and
current trajectory point, not all overlapping history/future windows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional, Union

import torch
import yaml
from tqdm import tqdm

from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import get_dataset_dir
from vla_bc_policy_temporal.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy_temporal.dataset.utility import IMAGE_OBS_KEY, VECTION_OBS_KEY


class RunningStats:
    """Stable streaming population mean/std/min/max; feature- or channel-wise."""

    def __init__(self, eps: float = 1e-12):
        if eps < 0:
            raise ValueError("eps must be nonnegative")
        self.eps = eps
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None
        self.min: Optional[torch.Tensor] = None
        self.max: Optional[torch.Tensor] = None

    def _update_rows(self, rows: torch.Tensor) -> None:
        # rows [N, D]: N observations per feature, in double precision.
        rows = torch.as_tensor(rows, dtype=torch.float64, device="cpu")
        if rows.ndim != 2 or rows.shape[0] == 0:
            raise ValueError(f"Expected a nonempty [N, D] tensor, got {tuple(rows.shape)}")
        if not torch.isfinite(rows).all():
            raise ValueError("Non-finite value in statistics input")
        n = int(rows.shape[0])
        batch_mean = rows.mean(dim=0)
        batch_m2 = ((rows - batch_mean) ** 2).sum(dim=0)
        batch_min = rows.amin(dim=0)
        batch_max = rows.amax(dim=0)
        if self.count == 0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.min = batch_min
            self.max = batch_max
        else:
            assert self.mean is not None and self.m2 is not None
            assert self.min is not None and self.max is not None
            if self.mean.shape != batch_mean.shape:
                raise ValueError(f"Feature dimension changed: {self.mean.shape} -> {batch_mean.shape}")
            total = self.count + n
            delta = batch_mean - self.mean
            self.mean += delta * (n / total)
            self.m2 += batch_m2 + delta.square() * (self.count * n / total)
            self.min = torch.minimum(self.min, batch_min)
            self.max = torch.maximum(self.max, batch_max)
        self.count += n

    @torch.no_grad()
    def update_feature(self, x: torch.Tensor) -> None:
        """Features [B, D], [B, T, D] (T is pooled), or [B] scalar."""
        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim < 2:
            raise ValueError(f"Expected feature batch, got {tuple(x.shape)}")
        self._update_rows(x.reshape(-1, x.shape[-1]))

    @torch.no_grad()
    def update_channel(self, x: torch.Tensor) -> None:
        """Channel-first images [B,C,H,W] or [B,T,C,H,W]."""
        x = torch.as_tensor(x)
        if x.ndim not in (4, 5):
            raise ValueError(f"Expected [B,C,H,W] or [B,T,C,H,W], got {tuple(x.shape)}")
        c_axis = x.ndim - 3
        # [C, all other dimensions] -> [N,C]
        rows = x.movedim(c_axis, -1).reshape(-1, x.shape[c_axis])
        self._update_rows(rows)

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("RunningStats is empty")
        assert self.mean is not None and self.m2 is not None
        assert self.min is not None and self.max is not None
        std = (self.m2 / self.count + self.eps).sqrt()
        return {
            "count": self.count,
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
        }


@torch.no_grad()
def compute_pi0_lerobot_stats_from_datamodule(
    dm: Pi0LeRobotDataModule,
    output_path: Optional[Union[str, Path]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    max_batches: Optional[int] = None,
    compute_image_stats: bool = True,
    compute_traject_stats: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute *train-only* statistics from current (non-overlapping) frames.

    Each sample is ``(obs, actions, trajects, masks)``.  Avoid counting one
    frame T times merely because it appears in overlapping temporal windows.
    The train Subset is created by the datamodule's episode-level split.
    """
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None")
    dm.setup(stage="fit")
    loader = dm.stats_dataloader(batch_size=batch_size, num_workers=num_workers)
    vector_stats: dict[str, RunningStats] = {}
    image_stats: dict[str, RunningStats] = {}
    action_stats = RunningStats()
    traject_stats = RunningStats() if compute_traject_stats else None

    num_batches = 0
    num_samples_seen = 0
    iterator = tqdm(loader, desc="Computing train stats") if progress else loader
    for obs_batch, action_batch, traject_batch, masks in iterator:
        # Current step is *always* valid for an indexed training sample.
        if not torch.as_tensor(masks["obs_valid_mask"])[:, -1].all():
            raise ValueError("Invalid current observation in stats loader")
        if not torch.as_tensor(masks["action_valid_mask"])[:, 0].all():
            raise ValueError("Invalid current action in stats loader")
        current_vector = obs_batch[VECTION_OBS_KEY]
        for key, value in current_vector.items():
            vector_stats.setdefault(key, RunningStats()).update_feature(value[:, -1, :])
        if compute_image_stats:
            for key, value in obs_batch[IMAGE_OBS_KEY].items():
                image_stats.setdefault(key, RunningStats()).update_channel(value[:, -1])
        action_stats.update_feature(action_batch[:, 0, :])
        if traject_stats is not None:
            center = dm.dataset_config.traject_horizon[0]
            traj_mask = torch.as_tensor(masks["traject_valid_mask"])[:, center]
            if not traj_mask.all():
                raise ValueError("Invalid current trajectory point in stats loader")
            traject_stats.update_feature(traject_batch[:, center, :])

        num_batches += 1
        num_samples_seen += int(action_batch.shape[0])
        if max_batches is not None and num_batches >= max_batches:
            break

    if num_batches == 0:
        raise ValueError("Empty training split: cannot compute statistics")

    stats: dict[str, Any] = {
        "schema_version": 2,
        "repo_id": dm.repo_id,
        "split": {
            "val_ratio": dm.val_ratio if dm.is_split_val else None,
            "split_seed": dm.seed,
            "val_repo_id": dm.val_repo_id,
            "train_len": len(dm.train_dataset),  # type: ignore[arg-type]
            "val_len": len(dm.val_dataset) if dm.val_dataset is not None else 0, # type: ignore
            "train_episode_ids": dm.train_episode_ids,
            "val_episode_ids": dm.val_episode_ids,
            "num_samples_seen": num_samples_seen,
            "num_batches_seen": num_batches,
            "max_batches": max_batches,
            "counting_rule": "current_frame_only",
            "vec_obs_keys": list(dm.dataset_config.vec_obs_keys),
        },
        "obs": {
            VECTION_OBS_KEY: {key: value.to_dict() for key, value in vector_stats.items()},
            IMAGE_OBS_KEY: {key: value.to_dict() for key, value in image_stats.items()},
        },
        "action": action_stats.to_dict(),
    }
    if traject_stats is not None:
        stats["traject"] = traject_stats.to_dict()

    if output_path is None:
        # Do not tie this module to one particular LeRobot import/version.
        output_path = get_dataset_dir(dm.repo_id, dm.root) / "train_stats.yaml"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, allow_unicode=True, sort_keys=False)
    return stats

if __name__ == "__main__":
    import tyro
    from vla_bc_policy_temporal.dataset.camera_config import TianyiCameraConfig, MshabCameraConfig, to_raw_camera_config_dict

    def main(
        repo_id: str = "tianyi_test",
        output_path: str = "output/train_stats.yaml",
        batch_size: int = 512,
        num_workers: int = 8,
        max_batches: Optional[int] = None,

        is_mshab_repo: bool = False,

        is_split_val: bool = True,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> None:
        if is_mshab_repo:
            dm = Pi0LeRobotDataModule(
                repo_id = repo_id,
                dataset_config = {
                    "camera_config_dict": to_raw_camera_config_dict(MshabCameraConfig),
                    "vec_obs_keys": ["pi0_state", "qpos"],

                    "ep_key": "episode_idx",
                    "frame_key": "frame_idx",

                    "observation_horizon": 1,
                    "action_horizon": 1,
                    "traject_horizon": (0, 1)
                },
                batch_size = batch_size,
                num_workers = num_workers,

                is_split_val = is_split_val,
                val_ratio = val_ratio,
                seed = seed
            )
        else:
            dm = Pi0LeRobotDataModule(
                repo_id = repo_id,
                dataset_config = {
                    "camera_config_dict": to_raw_camera_config_dict(TianyiCameraConfig),
                    "vec_obs_keys": ["qpos", "odom_rel"],

                    "observation_horizon": 1,
                    "action_horizon": 1,
                    "traject_horizon": (0, 1)
                },
                batch_size = batch_size,
                num_workers = num_workers,

                is_split_val = is_split_val,
                val_ratio = val_ratio,
                seed = seed
            )
        compute_pi0_lerobot_stats_from_datamodule(
            dm, output_path = output_path, max_batches = max_batches,
        )

    tyro.cli(main)
