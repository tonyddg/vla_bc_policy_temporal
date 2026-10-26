"""Lightning behavior cloning policy: extractor -> decoder -> action-chunk head.

Dataset batch: (obs, actions, trajects, masks).
Forward returns NORMALIZED [B,T_action,D_action]; predict_action() denormalizes.
The YAML stats are read on first creation only. Stats and method metadata are
in persistent buffers/extra_state, so load_from_checkpoint() needs no YAML.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn, optim

from vla_bc_policy_temporal.dataset.normalizer import Normalizer, NormalizeMethod
from vla_bc_policy_temporal.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy_temporal.model.extractor import ExtractorDict, ExtractorType

from vla_bc_policy_temporal.model.action_head import ActionHeadType, ActionHeadDict
from vla_bc_policy_temporal.model.mlp import MlpType, MlpDict

from vla_bc_policy_temporal.model.utility import LRScheduleType, WarmupCosineLR


def _require_match(kwargs: dict[str, Any], key: str, value: Any) -> None:
    if key in kwargs and kwargs[key] != value:
        raise ValueError(f"{key}: configured {kwargs[key]!r}, DataModule reports {value!r}")
    kwargs[key] = deepcopy(value)


def _ranges(groups: Optional[Mapping[str, tuple[int, int]]], action_dim: int):
    if groups is None:
        return None
    result: dict[str, tuple[int, int]] = {}
    for key, indices in groups.items():
        start, end = (int(indices[0]), int(indices[1]))
        if not 0 <= start < end <= action_dim:
            raise ValueError(f"action_group[{key!r}] invalid for D={action_dim}: {indices}")
        result[key] = (start, end)
    return result

# 问题:
# 1. 模型各个特征提取器、主干网络、动作头的网络规模与类型如何选择？既有足够性能又能在 20ms 内完成预测？
# 2. 历史观测与参考轨迹观测的长度与序列编码器如何选择？是否需要跳跃间隔以将更长的视野输入网络？
# 3. 当前模型没有计入视觉的历史观测，是否需要加入视觉历史观测？
# 4. mask 的使用：特征提取部分是否需要加入 mask？/ 动作预测是否需要去除 mask 让模型学会预测停止动作？
# 5. 合适的 batch size / 学习率调度 / loss（包括动作头损失权重）
# 6. (improved) 是否需要加入 bet / choice policy

class BCPolicy(pl.LightningModule):
    def __init__(
        self,
        extractor_type: ExtractorType = "cur_img_temp_vec_traj",
        extractor_kwargs: Mapping[str, Any] | None = None,

        decoder_type: MlpType = "mlp",
        decoder_kwargs: Mapping[str, Any] | None = None,

        head_type: ActionHeadType = "single",
        head_kwargs: Mapping[str, Any] | None = None,

        action_group: Optional[Mapping[str, tuple[int, int]]] = None,
        multi_head_weight: Optional[Mapping[str, float]] = None,

        lr: float = 2e-4,
        is_use_adam: bool = True,
        weight_decay: float = 5e-6,
        sgd_momentum: float = 0.99,
        loss_type: Literal["mse", "l1", "smooth_l1"] = "mse",

        key_metrics: str = "RMSE",

        lr_schedule_type: LRScheduleType = "none",
        lr_schedule_kwargs: Mapping[str, Any] | None = None,

        stats_path: Optional[Union[str, Path]] = None,
        image_method: NormalizeMethod = "none",
        vector_method: NormalizeMethod = "standard",
        action_method: NormalizeMethod = "minmax",
        traject_method: NormalizeMethod = "standard",
        method_by_key: Optional[Mapping[str, NormalizeMethod]] = None,

        data_module: Optional[Pi0LeRobotDataModule] = None,
    ):
        super().__init__()
        extractor_args = deepcopy(dict(extractor_kwargs or {}))
        decoder_args = deepcopy(dict(decoder_kwargs or {}))
        head_args = deepcopy(dict(head_kwargs or {}))
        lr_args = deepcopy(dict(lr_schedule_kwargs or {}))
        if lr <= 0 or weight_decay < 0:
            raise ValueError("lr must be positive and weight_decay nonnegative")
        if loss_type not in {"mse", "l1", "smooth_l1"}:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        if key_metrics not in {"RMSE", "MAE", "MedAE"}:
            raise ValueError("key_metrics must be one of RMSE, MAE, MedAE")

        if data_module is not None:
            for name, value in (
                ("vec_obs_dim", data_module.vec_obs_dim),
                ("img_obs_dim", data_module.img_obs_dim),
                ("traj_dim", data_module.traject_dim),
                ("vec_horizon", data_module.observation_horizon),
                ("traj_horizon", tuple(data_module.dataset_config.traject_horizon)),
            ):
                _require_match(extractor_args, name, value)
            _require_match(head_args, "num_out_feats", int(data_module.output_dim))
            _require_match(head_args, "action_horizon", int(data_module.action_horizon))

        decoder_args.setdefault("num_out_feats", 256)  # latent width, NOT action width
        if int(decoder_args["num_out_feats"]) <= 0:
            raise ValueError("Decoder latent width must be positive")
        _require_match(head_args, "num_in_feats", int(decoder_args["num_out_feats"]))
        if "num_out_feats" not in head_args or "action_horizon" not in head_args:
            raise ValueError("Supply DataModule or explicit action dimensions and action_horizon")

        # Resolve & store all dataset-derived model parameters in Lightning
        # hparams. stats_path is intentionally NONE: a checkpoint restores the
        # normalizer via nn.Buffer without needing the external training YAML.
        self.save_hyperparameters({
            "extractor_type": extractor_type, "extractor_kwargs": deepcopy(extractor_args),
            "decoder_type": decoder_type, "decoder_kwargs": deepcopy(decoder_args),
            "head_type": head_type, "head_kwargs": deepcopy(head_args),
            "stats_path": None,
            "action_group": deepcopy(dict(action_group)) if action_group is not None else None,
            "multi_head_weight": deepcopy(dict(multi_head_weight)) if multi_head_weight is not None else None,
            "lr": lr, "is_use_adam": is_use_adam, "weight_decay": weight_decay,
            "sgd_momentum": sgd_momentum, "loss_type": loss_type,
            "key_metrics": key_metrics, "lr_schedule_type": lr_schedule_type,
            "lr_schedule_kwargs": lr_args, "image_method": image_method,
            "vector_method": vector_method, "action_method": action_method,
            "traject_method": traject_method,
            "method_by_key": deepcopy(dict(method_by_key)) if method_by_key else None,
        })

        if extractor_type not in ExtractorDict:
            raise ValueError(f"Unknown extractor: {extractor_type}")

        self.extractor = ExtractorDict[extractor_type](**extractor_args)
        _require_match(decoder_args, "num_in_feats", self.extractor.get_out_feat())
        if decoder_type not in MlpDict:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")
        if decoder_type == "res_mlp":
            decoder_args.setdefault("hidden_dim", max(128, int(decoder_args["num_out_feats"])))
        self.decoder = MlpDict[decoder_type](**decoder_args)

        if head_type not in ActionHeadDict:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.head = ActionHeadDict[head_type](**head_args)
        self.action_dim = int(head_args["num_out_feats"])
        self.action_horizon = int(head_args["action_horizon"])
        self.action_group = _ranges(action_group, self.action_dim)
        self.multi_head_weight = dict(multi_head_weight) if multi_head_weight is not None else None
        if self.multi_head_weight is not None:
            if not self.action_group:
                raise ValueError("multi_head_weight requires action_group")
            if set(self.multi_head_weight) - set(self.action_group):
                raise ValueError("Weights mention unknown action groups")
            covered = []
            for start, end in self.action_group.values():
                covered.extend(range(start, end))
            if sorted(covered) != list(range(self.action_dim)):
                raise ValueError("Weighted action groups must cover every dimension exactly once")
            if any(not 0 <= float(w) for w in self.multi_head_weight.values()):
                raise ValueError("Group weights must be nonnegative")

        self.lr, self.is_use_adam, self.weight_decay = lr, is_use_adam, weight_decay
        self.sgd_momentum, self.loss_type = sgd_momentum, loss_type
        self.lr_schedule_type, self.lr_schedule_kwargs = lr_schedule_type, lr_args
        self.key_metrics = key_metrics
        self.normalizer = Normalizer(
            stats_path=stats_path, image_method=image_method,
            vector_method=vector_method, action_method=action_method,
            traject_method=traject_method, method_by_key=method_by_key,
        )
        if data_module is not None:
            example_obs, _, example_traj, example_masks = data_module.get_random_batch(batch_size=1)
            self.example_input_array = {
                "obs": example_obs, "traject": example_traj, "masks": example_masks,
            }

    def forward(self, obs: Mapping[str, Mapping[str, torch.Tensor]],
                traject: torch.Tensor | None = None,
                masks: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        obs_normalized = self.normalizer.normalize_obs(obs)
        traj_normalized = self.normalizer.normalize_traject(traject) if traject is not None else None
        features = self.extractor(obs_normalized, traj_normalized, masks)
        latent = self.decoder(features)
        return self.head(latent)

    @torch.no_grad()
    def predict_action(self, obs: Mapping[str, Mapping[str, torch.Tensor]],
                       traject: torch.Tensor | None = None,
                       masks: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        return self.normalizer.denormalize_action(self(obs, traject, masks))

    def _element_error(self, prediction: torch.Tensor, target: torch.Tensor):
        if self.loss_type == "mse":
            return (prediction - target).square()
        if self.loss_type == "l1":
            return (prediction - target).abs()
        return F.smooth_l1_loss(prediction, target, reduction="none")

    def compute_action_loss(self, prediction: torch.Tensor, target: torch.Tensor,
                            valid_mask: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError(f"Action shape mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}")
        if prediction.shape[1:] != (self.action_horizon, self.action_dim):
            raise ValueError("Action tensor does not match policy action horizon / dimension")
        if valid_mask.shape != prediction.shape[:2]:
            raise ValueError(f"action_valid_mask expected {tuple(prediction.shape[:2])}")
        
        valid = valid_mask.to(device=prediction.device, dtype=torch.bool)
        count = valid.sum()
        if count.item() == 0:
            raise ValueError("No valid action targets in batch")
        error = self._element_error(prediction, target)
        if self.multi_head_weight is None:
            return (error * valid.unsqueeze(-1)).sum() / (count * self.action_dim)
        loss = error.new_zeros(())
        for name, (start, end) in self.action_group.items():  # type: ignore[union-attr]
            group_loss = (error[:, :, start:end] * valid.unsqueeze(-1)).sum() / (count * (end-start))
            loss = loss + group_loss * float(self.multi_head_weight.get(name, 1.0)) * ((end-start)/self.action_dim)
        return loss

    @torch.no_grad()
    def _metrics(self, prediction: torch.Tensor, target: torch.Tensor,
                 valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        valid = valid_mask.to(device=prediction.device, dtype=torch.bool)
        errors = (prediction.detach() - target.detach()).float()
        values = errors[valid].reshape(-1)
        abs_values = values.abs()
        metrics = {
            "MAE": abs_values.mean(),
            "RMSE": values.square().mean().sqrt(),
            "MedAE": abs_values.median(),
        }
        for name, (start, end) in (self.action_group or {}).items():
            group_values = errors[:, :, start:end][valid].reshape(-1)
            metrics[f"{name}_MAE"] = group_values.abs().mean()
            metrics[f"{name}_RMSE"] = group_values.square().mean().sqrt()
        return metrics

    def get_loss(self, obs, target_action_norm, traject, masks):
        pred_action_norm = self(obs, traject, masks)
        return self.compute_action_loss(pred_action_norm, target_action_norm,
                                        masks["action_valid_mask"]), pred_action_norm

    def _shared_step(self, batch, phase: str):
        obs, actions, trajects, masks = batch
        target_normalized = self.normalizer.normalize_action(actions)
        loss, predicted = self.get_loss(obs, target_normalized, trajects, masks)
        batch_size = actions.shape[0]
        self.log(f"{phase}_loss", loss, batch_size=batch_size,
                 on_step=phase == "train", on_epoch=True,
                 prog_bar=phase == "train")
        for key, val in self._metrics(predicted, target_normalized,
                                      masks["action_valid_mask"]).items():
            self.log(f"{phase}_{key}", val, batch_size=batch_size,
                     on_step=False, on_epoch=True,
                     prog_bar=phase == "val" and key == self.key_metrics)
        return loss

    def training_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx: int):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        if self.is_use_adam:
            optimizer = optim.AdamW(
                self.parameters(), 
                lr=self.lr, 
                weight_decay=self.weight_decay
            )
        else:
            optimizer = optim.SGD(
                self.parameters(), 
                lr=self.lr,
                momentum=self.sgd_momentum, 
                weight_decay=self.weight_decay
            )

        args = dict(self.lr_schedule_kwargs)
        if self.lr_schedule_type == "none":
            return optimizer
        if self.lr_schedule_type == "warmup_cos":
            args.setdefault("total_steps", int(self.trainer.estimated_stepping_batches))
            scheduler = WarmupCosineLR(optimizer=optimizer, **args)
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler, "interval": "step", "frequency": 1}}
        if self.lr_schedule_type == "reduce":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **args)
            return {"optimizer": optimizer, "lr_scheduler": {
                "scheduler": scheduler, "monitor": f"val_{self.key_metrics}",
                "interval": "epoch", "frequency": 1}}
        raise ValueError(f"Unknown lr_schedule_type: {self.lr_schedule_type}")


BCPolicyModule = BCPolicy
