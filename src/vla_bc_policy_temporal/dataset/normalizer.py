"""YAML-backed, checkpoint-restorable nested observation normalizer.

First training run: ``Normalizer(stats_path='train_stats.yaml')``.
Checkpoint restore: ``Normalizer().load_state_dict(saved_state_dict)``.
All numerical statistics are persistent ``nn.Buffer`` values (not plain attrs).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import torch
import yaml
from torch import Tensor, nn

from vla_bc_policy_temporal.dataset.utility import IMAGE_OBS_KEY, VECTION_OBS_KEY

NormalizeMethod = Literal["none", "standard", "minmax", "minmax_0_1"]
_METHODS = {"none", "standard", "minmax", "minmax_0_1"}
_STAT_NAMES = ("mean", "std", "min", "max")


def _encode_key(key: str) -> str:
    """Reversible PyTorch ModuleDict key, even when dataset keys contain dots."""
    return "key_" + key.encode("utf-8").hex()


class _StatBuffers(nn.Module):
    def __init__(self, stats: Mapping[str, Any]):
        super().__init__()
        tensors = {}
        for name in _STAT_NAMES:
            if name not in stats:
                raise ValueError(f"Missing statistics field {name!r}")
            tensor = torch.as_tensor(stats[name], dtype=torch.float32).flatten()
            if tensor.numel() == 0 or not torch.isfinite(tensor).all():
                raise ValueError(f"Empty or non-finite statistics field {name!r}")
            tensors[name] = tensor
        width = tensors["mean"].numel()
        if any(t.numel() != width for t in tensors.values()):
            raise ValueError("mean/std/min/max must have identical feature counts")
        if (tensors["std"] < 0).any():
            raise ValueError("std must be nonnegative")
        if (tensors["min"] > tensors["max"]).any():
            raise ValueError("min cannot be greater than max")
        for name, value in tensors.items():
            setattr(self, name, nn.Buffer(value.clone(), persistent=True))


class Normalizer(nn.Module):
    """Normalize nested observations without concatenating vector features.

    Expected obs schema::
        {
            "vector": {"qpos": [B,T,D], "qvel": [B,T,D2]},
            "image": {"head": [B,T,C,H,W]},
        }

    YAML schema_version=2, with obs.vector.<key>, obs.image.<key>,
    action and optional traject, each containing mean/std/min/max.
    The module can be reconstructed from ``state_dict`` *without the YAML*.
    """

    def __init__(
        self,
        stats_path: Optional[Union[str, Path]] = None,
        image_method: NormalizeMethod = "none",
        vector_method: NormalizeMethod = "standard",
        action_method: NormalizeMethod = "minmax",
        traject_method: NormalizeMethod = "standard",
        # 通过 obs.vector.{key} / obs.image.{key} 指定标准化方法, 指定时具有最高优先级
        method_by_key: Optional[Mapping[str, NormalizeMethod]] = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.eps = float(eps)
        self.image_method = image_method
        self.vector_method = vector_method
        self.action_method = action_method
        self.traject_method = traject_method
        self.method_by_key = dict(method_by_key or {})
        self._validate_methods()

        self.vector_stats = nn.ModuleDict()
        self.image_stats = nn.ModuleDict()
        self.action_stats = nn.ModuleDict()
        self.traject_stats = nn.ModuleDict()
        
        # self.register_buffer("stats_ready", nn.Buffer(torch.tensor(False), persistent=True))
        self.stats_ready  = nn.Buffer(torch.tensor(False), persistent=True)
        
        self._vector_keys: list[str] = []
        self._image_keys: list[str] = []
        if stats_path is not None:
            self.load_stats_yaml(stats_path)

    def _validate_methods(self) -> None:
        methods = [self.image_method, self.vector_method, self.action_method,
                   self.traject_method, *self.method_by_key.values()]
        if not all(m in _METHODS for m in methods):
            raise ValueError(f"Invalid normalization method in {methods}; choose {_METHODS}")

    @torch.no_grad()
    def load_stats_yaml(self, path: Union[str, Path]) -> None:
        """Load once, into persistent PyTorch buffers; usually only on first run."""
        if bool(self.stats_ready.item()):
            raise RuntimeError("Stats already initialized; create a new Normalizer to reload")
        with Path(path).open("r", encoding="utf-8") as f:
            stats = yaml.safe_load(f)
        if not isinstance(stats, Mapping) or stats.get("schema_version") != 2:
            raise ValueError("Expected train_stats.yaml with schema_version: 2")
        obs = stats.get("obs")
        if not isinstance(obs, Mapping):
            raise ValueError("Stats missing 'obs' mapping")
        vectors = obs.get(VECTION_OBS_KEY, {})
        images = obs.get(IMAGE_OBS_KEY, {})
        if not isinstance(vectors, Mapping) or not isinstance(images, Mapping):
            raise ValueError("obs.vector and obs.image must be dictionaries")
        if not isinstance(stats.get("action"), Mapping):
            raise ValueError("Stats missing 'action' mapping")
        for key, values in vectors.items():
            self.vector_stats[_encode_key(key)] = _StatBuffers(values)
        for key, values in images.items():
            self.image_stats[_encode_key(key)] = _StatBuffers(values)
        self.action_stats["single"] = _StatBuffers(stats["action"])
        if "traject" in stats:
            self.traject_stats["single"] = _StatBuffers(stats["traject"])
        self._vector_keys = list(vectors)
        self._image_keys = list(images)
        self.stats_ready.fill_(True)

    def get_extra_state(self) -> dict[str, Any]:
        """Pure-Python metadata needed to rebuild ModuleDict keys on reload."""
        return {
            "version": 2,
            "vector_keys": list(self._vector_keys),
            "image_keys": list(self._image_keys),
            "image_method": self.image_method,
            "vector_method": self.vector_method,
            "action_method": self.action_method,
            "traject_method": self.traject_method,
            "method_by_key": dict(self.method_by_key),
            "eps": self.eps,
        }

    def set_extra_state(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 2:
            raise ValueError("Unsupported normalizer checkpoint schema")
        self._vector_keys = list(state["vector_keys"])
        self._image_keys = list(state["image_keys"])
        self.image_method = state["image_method"]
        self.vector_method = state["vector_method"]
        self.action_method = state["action_method"]
        self.traject_method = state["traject_method"]
        self.method_by_key = dict(state["method_by_key"])
        self.eps = float(state["eps"])
        self._validate_methods()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # PyTorch calls parent before children: pre-create child buffer modules.
        # Otherwise a newly instantiated Normalizer() cannot load its own stats.
        metadata = state_dict.get(prefix + "_extra_state")
        if metadata is not None:
            for category, keys, module in (
                ("vector_stats", metadata["vector_keys"], self.vector_stats),
                ("image_stats", metadata["image_keys"], self.image_stats),
            ):
                for key in keys:
                    encoded = _encode_key(key)
                    if encoded not in module:
                        self._restore_stat_module(module, encoded, state_dict, prefix + category + ".")
            for category, module in (("action_stats", self.action_stats),
                                     ("traject_stats", self.traject_stats)):
                name = prefix + category + ".single.mean"
                if name in state_dict and "single" not in module:
                    self._restore_stat_module(module, "single", state_dict, prefix + category + ".")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _restore_stat_module(module, key: str, state_dict, prefix: str) -> None:
        values = {}
        for name in _STAT_NAMES:
            lookup = prefix + key + "." + name
            if lookup not in state_dict:
                raise ValueError(f"Checkpoint missing {lookup}")
            values[name] = state_dict[lookup]
        module[key] = _StatBuffers(values)

    def _check_ready(self) -> None:
        if not bool(self.stats_ready.item()):
            raise RuntimeError("Stats not initialized: supply stats_path or load_state_dict")

    def _apply_tensor(self, x: Tensor, buffers: _StatBuffers, method: NormalizeMethod,
                      is_image: bool, inverse: bool) -> Tensor:
        if method == "none":
            return x
        if not torch.is_floating_point(x):
            x = x.float()
        axis = x.ndim - 3 if is_image else x.ndim - 1
        if is_image and x.ndim < 3:
            raise ValueError(f"Expected channel-first image, got shape {tuple(x.shape)}")
        def shaped(name: str) -> Tensor:
            value = getattr(buffers, name).to(dtype=x.dtype, device=x.device)
            if x.shape[axis] != value.numel():
                raise ValueError(f"Statistic {name} has {value.numel()} elements, got {tuple(x.shape)}")
            shape = [1] * x.ndim
            shape[axis] = value.numel()
            return value.reshape(shape)
        if method == "standard":
            mu, sigma = shaped("mean"), shaped("std").clamp_min(self.eps)
            return x * sigma + mu if inverse else (x - mu) / sigma
        low, high = shaped("min"), shaped("max")
        denom = (high - low).clamp_min(self.eps)
        if method == "minmax":
            return (x + 1) * 0.5 * (high - low) + low if inverse else 2 * (x - low) / denom - 1
        if method == "minmax_0_1":
            return x * (high - low) + low if inverse else (x - low) / denom
        raise ValueError(f"Invalid normalization method: {method}")

    def _method(self, path: str, default: NormalizeMethod, override: Optional[NormalizeMethod]) -> NormalizeMethod:
        method = override if override is not None else self.method_by_key.get(path, default)
        if method not in _METHODS:
            raise ValueError(f"Invalid method {method!r}")
        return method # type: ignore

    def _handle(self, value: Tensor, module: nn.ModuleDict, key: str,
                path: str, default: NormalizeMethod, method: Optional[NormalizeMethod],
                is_image: bool, inverse: bool) -> Tensor:
        selected = self._method(path, default, method)
        if selected == "none":
            return value
        encoded = "single" if key == "single" else _encode_key(key)
        if encoded not in module:
            raise KeyError(f"No statistics for {path!r}")
        return self._apply_tensor(value, module[encoded], selected, is_image, inverse) # type: ignore

    def _observation(self, obs: Mapping[str, Mapping[str, Tensor]],
                     method: Optional[NormalizeMethod], inverse: bool):
        self._check_ready()
        if VECTION_OBS_KEY not in obs or IMAGE_OBS_KEY not in obs:
            raise KeyError("Expected nested obs={'vector': {...}, 'image': {...}}")
        out: dict[str, dict[str, Tensor]] = {VECTION_OBS_KEY: {}, IMAGE_OBS_KEY: {}}
        
        for key, value in obs[VECTION_OBS_KEY].items():
            out[VECTION_OBS_KEY][key] = self._handle(
                value, self.vector_stats, key, f"obs.vector.{key}",
                self.vector_method, method, False, inverse, # type: ignore
            )
        for key, value in obs[IMAGE_OBS_KEY].items():
            out[IMAGE_OBS_KEY][key] = self._handle(
                value, self.image_stats, key, f"obs.image.{key}",
                self.image_method, method, True, inverse, # type: ignore
            )
        return out

    def normalize_obs(self, obs: Mapping[str, Mapping[str, Tensor]],
                      method: Optional[NormalizeMethod] = None):
        return self._observation(obs, method, inverse=False)

    def denormalize_obs(self, obs: Mapping[str, Mapping[str, Tensor]],
                        method: Optional[NormalizeMethod] = None):
        return self._observation(obs, method, inverse=True)

    def normalize_action(self, action: Tensor, method: Optional[NormalizeMethod] = None):
        self._check_ready()
        return self._handle(action, self.action_stats, "single", "action",
                            self.action_method, method, False, False) # type: ignore

    def denormalize_action(self, action: Tensor, method: Optional[NormalizeMethod] = None):
        self._check_ready()
        return self._handle(action, self.action_stats, "single", "action",
                            self.action_method, method, False, True) # type: ignore

    def normalize_traject(self, traject: Tensor, method: Optional[NormalizeMethod] = None):
        self._check_ready()
        return self._handle(traject, self.traject_stats, "single", "traject",
                            self.traject_method, method, False, False) # type: ignore

    def denormalize_traject(self, traject: Tensor, method: Optional[NormalizeMethod] = None):
        self._check_ready()
        return self._handle(traject, self.traject_stats, "single", "traject",
                            self.traject_method, method, False, True) # type: ignore

    def forward(self, obs: Mapping[str, Mapping[str, Tensor]],
                action: Optional[Tensor] = None,
                method: Optional[NormalizeMethod] = None):
        result = self.normalize_obs(obs, method)
        return result if action is None else (result, self.normalize_action(action, method))
