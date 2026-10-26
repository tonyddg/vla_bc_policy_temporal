"""LeRobot DataModule with persistent, episode-level train/validation splits.

The split file is saved to <resolved LeRobot repo>/meta/train_val_split.json.
It stores episode IDs instead of individual frame indices. Frame indices are
reconstructed from the metadata column each time (without decoding images).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import torch
from lightning import pytorch as pl
from torch.utils.data import DataLoader, Subset, Dataset

from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import (
    Pi0LeRobotDataset,
    Pi0LeRobotDatasetConfig,
)
from vla_bc_policy_temporal.dataset.utility import (
    get_random_pi0_lerobot_batch,
    random_split_indices,
)

SPLIT_SCHEMA_VERSION = 1
DEFAULT_SPLIT_RELATIVE_PATH = Path("meta") / "train_val_split.json"

# Each segment is (episode_id, start_frame_index_in_dataset, exclusive_end).
EpisodeSegment = tuple[int, int, int]

def _episode_segments(dataset: Pi0LeRobotDataset) -> list[EpisodeSegment]:
    """Read only episode IDs, never full image-bearing dataset samples.

    LeRobot datasets normally place all frames from an episode in one
    contiguous range. Check that assumption explicitly: otherwise a
    frame-level Subset built from these ranges could mix episodes.
    """
    episode_key = dataset.ep_key
    try:
        column = dataset.ds.hf_dataset[episode_key]
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"Cannot read the episode column {episode_key!r} from the "
            "underlying LeRobot dataset."
        ) from exc

    segments: list[EpisodeSegment] = []
    seen_ids: set[int] = set()
    prev_episode: Optional[int] = None
    start = 0
    row_count = 0

    for row_count, value in enumerate(column, start=1):
        ep = int(torch.as_tensor(value).item())
        index = row_count - 1
        if ep != prev_episode:
            if prev_episode is not None:
                segments.append((prev_episode, start, index))
            if ep in seen_ids:
                raise ValueError(
                    f"Episode {ep} appears in noncontiguous frame ranges. "
                    "Please repair/reorder the LeRobot episode_index column."
                )
            seen_ids.add(ep)
            prev_episode = ep
            start = index

    if prev_episode is not None:
        segments.append((prev_episode, start, row_count))

    if row_count != len(dataset):
        raise RuntimeError(
            f"Episode column length {row_count} differs from dataset length "
            f"{len(dataset)}."
        )
    if not segments:
        raise ValueError("The LeRobot dataset is empty.")
    return segments

def _dataset_fingerprint(segments: list[EpisodeSegment]) -> str:
    """Hash episode IDs and their exact frame ranges (not file timestamps)."""
    payload = json.dumps(segments, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _atomic_dump_json(path: Path, data: Mapping[str, Any]) -> None:
    """Replace a complete JSON document atomically, including on DDP ranks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            tmp_name = file.name
            json.dump(data, file, indent=2, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass
        
class Pi0LeRobotDataModule(pl.LightningDataModule):
    def __init__(
        self,

        repo_id: str, 
        dataset_config: Mapping,

        root: Optional[Union[str, Path]] = None,

        batch_size: int = 512,
        num_workers: int = 8,

        # 使用独立的验证集
        val_repo_id: Optional[str] = None, 
        # 没有独立验证集时将从训练集划分 (没有提供独立验证集时有效)
        is_split_val: bool = False,
        val_ratio: float = 0.1,
        seed: int = 42,

        detach_h5_path: Optional[Union[str, Path]] = None,
        val_detach_h5_path: Optional[Union[str, Path]] = None,

        # 使用保守的 dataloader 配置
        debug_config: bool = False,

        split_path: Optional[Union[str, Path]] = None,
        force_resplit: bool = False,
    ) -> None:
        super().__init__()
        
        if not repo_id:
            raise ValueError("repo_id must be a nonempty string.")
        if val_repo_id is not None and is_split_val:
            raise ValueError(
                "val_repo_id and is_split_val=True are mutually exclusive."
            )
        if is_split_val and (not math.isfinite(val_ratio) or not 0 < val_ratio < 1):
            raise ValueError(
                "With is_split_val=True, val_ratio must be strictly between 0 and 1."
            )
        if batch_size <= 0 or num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative.")
        if type(seed) is not int:
            raise TypeError("seed must be an integer.")

        self.save_hyperparameters()
        self.repo_id = repo_id
        self.root = root
        self.dataset_config = Pi0LeRobotDatasetConfig.from_dict_config(dataset_config)
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_repo_id = val_repo_id
        self.is_split_val = is_split_val
        self.val_ratio = float(val_ratio)
        self.seed = int(seed)
        self.debug_config = debug_config
        
        self.detach_h5_path = detach_h5_path
        self.val_detach_h5_path = val_detach_h5_path
        
        self.full_dataset: Optional[Pi0LeRobotDataset] = None
        self.train_dataset: Optional[Union[Pi0LeRobotDataset, Subset]] = None
        self.val_dataset: Optional[Union[Pi0LeRobotDataset, Subset]] = None

        # Populated by setup() when an in-repo episode split is active.
        self.split_path = Path(split_path) if split_path is not None else None
        self.force_resplit = force_resplit
        self.split_file_path: Optional[Path] = None
        self.train_episode_ids: Optional[list[int]] = None
        self.val_episode_ids: Optional[list[int]] = None

        # 获取样本输入尺寸
        # If any features are from detach.h5 and absent in meta/info.json,
        # shape inference falls back to a real Dataset sample.
        try:
            dims = self.dataset_config.get_data_feature_dim(repo_id, root)
        except (KeyError, FileNotFoundError):
            dataset = self._make_dataset(repo_id, detach_h5_path)
            dims = dataset.get_sample_size()
            self.full_dataset = dataset
        self.output_dim, self.traject_dim, self.vec_obs_dim, self.img_obs_dim = dims
        self.action_horizon, self.traject_horizon, self.observation_horizon = self.dataset_config.get_horizon()

    def _make_dataset(self, repo_id: str, detach_path=None) -> Pi0LeRobotDataset:
        return Pi0LeRobotDataset(
            repo_id, config=self.dataset_config, root=self.root,
            detach_h5_path=detach_path,
        )
    
    def get_dataspec(self):
        return (self.output_dim, self.traject_dim, self.vec_obs_dim, self.img_obs_dim)

    def get_random_batch(self, batch_size: int = 1):
        return get_random_pi0_lerobot_batch(
            output_dim = self.output_dim, 
            traject_dim = self.traject_dim, 
            vec_obs_dim = self.vec_obs_dim,
            img_obs_dim = self.img_obs_dim,
            action_horizon = self.action_horizon, 
            traject_horizon = self.traject_horizon, 
            observation_horizon = self.observation_horizon,
            batch_size = batch_size
        )

    def _resolved_split_file(self, dataset: Pi0LeRobotDataset) -> Path:
        repo_dir = Path(dataset.dataset_dir_path)
        if self.split_path is None:
            return repo_dir / DEFAULT_SPLIT_RELATIVE_PATH
        if self.split_path.is_absolute():
            return self.split_path
        return repo_dir / self.split_path

    def _create_manifest(
        self, segments: list[EpisodeSegment]
    ) -> dict[str, Any]:
        n_episodes = len(segments)
        if n_episodes < 2:
            raise ValueError(
                "At least two episodes are required for a nonempty train/val split."
            )

        # Prefer one val episode over an empty validation set when the
        # dataset is small and val_ratio * n_episodes rounds to zero.
        n_val = max(
            1, min(n_episodes - 1, math.floor(n_episodes * self.val_ratio + 0.5))
        )
        val_positions, train_positions = random_split_indices(
            n_episodes, n_val, seed=self.seed
        )
        val_set = set(val_positions)
        train_set = set(train_positions)
        # Store IDs in dataset order regardless of RNG sampling order.
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "repo_id": self.repo_id,
            "dataset_length": len(self.full_dataset),  # type: ignore[arg-type]
            "episode_count": n_episodes,
            "episode_fingerprint": _dataset_fingerprint(segments),
            "seed": self.seed,
            "val_ratio": self.val_ratio,
            "train_episode_ids": [
                ep for pos, (ep, _, _) in enumerate(segments)
                if pos in train_set
            ],
            "val_episode_ids": [
                ep for pos, (ep, _, _) in enumerate(segments)
                if pos in val_set
            ],
        }

    def _validate_manifest(
        self, manifest: Any, segments: list[EpisodeSegment], path: Path
    ) -> tuple[list[int], list[int]]:
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid episode split manifest: {path}")

        expected = {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "repo_id": self.repo_id,
            "dataset_length": len(self.full_dataset),  # type: ignore[arg-type]
            "episode_count": len(segments),
            "episode_fingerprint": _dataset_fingerprint(segments),
            "seed": self.seed,
        }
        for name, value in expected.items():
            if manifest.get(name) != value:
                raise ValueError(
                    f"Stored split {path} has incompatible {name}: "
                    f"{manifest.get(name)!r} != {value!r}. "
                    "Set force_resplit=True to replace it intentionally, "
                    "or use a different split_path."
                )

        saved_ratio = manifest.get("val_ratio", self.val_ratio)
        if (
            type(saved_ratio) not in (int, float)
            or not math.isclose(saved_ratio, self.val_ratio, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Stored split {path} has val_ratio={saved_ratio!r}, "
                f"but the DataModule requests {self.val_ratio!r}. "
                "Set force_resplit=True or use another split_path."
            )

        train_ids = manifest.get("train_episode_ids")
        val_ids = manifest.get("val_episode_ids")
        if (
            not isinstance(train_ids, list)
            or not isinstance(val_ids, list)
            or not train_ids
            or not val_ids
            or any(type(ep) is not int for ep in train_ids + val_ids)
        ):
            raise ValueError(f"Stored split {path} has invalid episode ID lists.")

        all_ids = {ep for ep, _, _ in segments}
        train_set, val_set = set(train_ids), set(val_ids)
        if (
            len(train_set) != len(train_ids)
            or len(val_set) != len(val_ids)
            or train_set & val_set
            or train_set | val_set != all_ids
        ):
            raise ValueError(
                f"Stored split {path} contains duplicate, overlapping, "
                "missing, or unknown episode IDs. Set force_resplit=True."
            )
        return train_ids, val_ids

    def _load_or_create_split(
        self, segments: list[EpisodeSegment], path: Path
    ) -> tuple[list[int], list[int]]:
        if path.exists() and not self.force_resplit:
            try:
                with path.open("r", encoding="utf-8") as file:
                    manifest = json.load(file)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot read episode split at {path}. "
                    "Inspect the file or set force_resplit=True."
                ) from exc
        else:
            manifest = self._create_manifest(segments)
            try:
                _atomic_dump_json(path, manifest)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot save episode split at {path}; "
                    "the LeRobot repo directory must be writable. "
                    "Alternatively specify a writable split_path."
                ) from exc

        # Always validate even after creating, and again on a subsequent
        # process/rank. os.replace prevents readers seeing partial JSON.
        return self._validate_manifest(manifest, segments, path)

    def setup(self, stage: Optional[str] = None) -> None:
        # Lightning may call setup("fit"), setup("validate"), etc. repeatedly.
        if self.train_dataset is not None and (
            not self.is_split_val or self.val_dataset is not None
        ):
            return

        if self.val_repo_id is not None:
            self.full_dataset = self.full_dataset or self._make_dataset(self.repo_id, self.detach_h5_path)
            self.train_dataset = self.full_dataset
            self.val_dataset = self._make_dataset(self.val_repo_id, self.val_detach_h5_path)
            return

        if not self.is_split_val:
            self.full_dataset = self.full_dataset or self._make_dataset(self.repo_id, self.detach_h5_path)
            self.train_dataset = self.full_dataset
            self.val_dataset = None
            return

        self.full_dataset = self.full_dataset or self._make_dataset(self.repo_id, self.detach_h5_path)
        segments = _episode_segments(self.full_dataset)
        self.split_file_path = self._resolved_split_file(self.full_dataset)

        train_ids, val_ids = self._load_or_create_split(
            segments, self.split_file_path
        )
        train_set, val_set = set(train_ids), set(val_ids)

        train_indices: list[int] = []
        val_indices: list[int] = []
        for episode_id, start, end in segments:
            if episode_id in train_set:
                train_indices.extend(range(start, end))
            elif episode_id in val_set:
                val_indices.extend(range(start, end))
            else:
                raise AssertionError(f"Episode {episode_id} was not assigned.")

        if not train_indices or not val_indices:
            raise RuntimeError("Train and validation frame subsets must be nonempty.")

        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.val_dataset = Subset(self.full_dataset, val_indices)
        self.train_episode_ids = train_ids
        self.val_episode_ids = val_ids

    def _loader(self, dataset: Dataset, *, shuffle: bool, drop_last: bool,
                batch_size: Optional[int] = None, num_workers: Optional[int] = None,
                stats: bool = False) -> DataLoader:
        batch_size = self.batch_size if batch_size is None else batch_size
        workers = self.num_workers if num_workers is None else num_workers
        if self.debug_config and not stats:
            workers = 0
        if batch_size < 1 or workers < 0:
            raise ValueError("Invalid dataloader batch_size or num_workers")
        return DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            num_workers = workers,
            pin_memory = (not self.debug_config and not stats),
            persistent_workers = workers > 0 and not stats,
            prefetch_factor = 2 if workers > 0 else None,
            drop_last = drop_last,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before train_dataloader()")
        # Avoid silently producing zero batches for a small training split.
        return self._loader(
            self.train_dataset, shuffle = True,
            drop_last = len(self.train_dataset) >= self.batch_size, # type: ignore
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("No validation dataset; configure is_split_val or val_repo_id")
        return self._loader(self.val_dataset, shuffle = False, drop_last = False)

    def stats_dataloader(self, batch_size: Optional[int] = None,
                         num_workers: Optional[int] = None) -> DataLoader:
        '''
        非 lightning 标准实现, 用于获取统计量
        '''
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before stats_dataloader()")
        return self._loader(self.train_dataset, shuffle = False, drop_last = False,
                            batch_size = batch_size, num_workers = num_workers, stats = True)
