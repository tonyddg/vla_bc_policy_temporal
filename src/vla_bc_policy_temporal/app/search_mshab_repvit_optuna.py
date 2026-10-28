"""Optuna LR / weight-decay / batch-size search for MS-HAB RepViT single-step BC.

Add this NEW file to:
    vla_bc_policy_temporal/app/search_mshab_repvit_optuna.py

Requires the previously added RepViT files, plus:
    pip install optuna 'timm>=1.0.15' tyro lightning tensorboard

Usage (from the installed project environment):
    python -m vla_bc_policy_temporal.app.search_mshab_repvit_optuna \
        --n-trials 15 --max-epochs 100 --num-workers 16

To resume another 10 trials from the same SQLite study:
    python -m vla_bc_policy_temporal.app.search_mshab_repvit_optuna \
        --n-trials 10 --max-epochs 100 --num-workers 16

Only three hyperparameters are searched:
    lr            log-uniform [1e-4, 5e-4]
    weight_decay  log-uniform [1e-5, 1e-2]
    batch_size    categorical {64, 128}

Objective: minimum *epoch-level* validation MAE (normalized single-step
10D whole-body action), not train loss or last-epoch validation MAE.
The same episode split, model architecture, seed, optimizer, cosine schedule,
loss weights, and normalization are used for all trials. Batch size changes
steps per epoch (and therefore total optimizer updates), as in normal training.

By default trials do not save model weights to avoid disk use during search.
The SQLite database, per-trial TensorBoard runs, trial_results.csv, and
best_params.json persist in --output-dir. Retrain the selected configuration
with the existing training entry to obtain a final checkpoint.
"""
from __future__ import annotations

import csv
import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import optuna
import torch
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from vla_bc_policy_temporal.dataset.camera_config import MshabCameraConfig
from vla_bc_policy_temporal.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import Pi0LeRobotDatasetConfig
from vla_bc_policy_temporal.policy.repvit_history_single_step_policy import (
    EXTRACTOR_KEY,
    IMAGE_KEY,
    REFERENCE_KEY,
    STATE_KEY,
    RepViTHistorySingleStepPolicy,
)


class BestValidationReporter(Callback):
    """Observe reduced val_* metrics once per epoch and optionally prune."""

    def __init__(self, trial: optuna.Trial):
        super().__init__()
        self.trial = trial
        self.best_mae = math.inf
        self.best_epoch = -1
        self.best_step = -1
        self.best_group_metrics: dict[str, float] = {}

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        if "val_MAE" not in metrics:
            raise RuntimeError("Expected val_MAE from BCPolicy.validation_step().")
        mae = float(metrics["val_MAE"].detach().cpu())
        if not math.isfinite(mae):
            raise optuna.TrialPruned("Non-finite validation MAE")

        epoch = int(trainer.current_epoch)
        self.trial.report(mae, step=epoch)
        if mae < self.best_mae:
            self.best_mae = mae
            self.best_epoch = epoch
            self.best_step = int(trainer.global_step)
            self.best_group_metrics = {}
            for name in ("val_loss", "val_joint_MAE", "val_torso_MAE", "val_base_MAE"):
                if name in metrics:
                    self.best_group_metrics[name] = float(metrics[name].detach().cpu())
            self.trial.set_user_attr("best_epoch", epoch)
            self.trial.set_user_attr("best_global_step", self.best_step)
            self.trial.set_user_attr("best_val_MAE", mae)
            for name, value in self.best_group_metrics.items():
                self.trial.set_user_attr(f"at_best_{name}", value)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Pruned after epoch {epoch + 1}, current val_MAE={mae:.6f}, "
                f"best val_MAE={self.best_mae:.6f}"
            )


def make_datamodule(
    repo_id: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    val_ratio: float,
    split_path: str | None,
) -> Pi0LeRobotDataModule:
    cfg = Pi0LeRobotDatasetConfig(
        camera_config_dict=MshabCameraConfig,
        vec_obs_keys=["qpos", "pi0_state"],
        observation_horizon=5,
        padding_obs_zero_keys=["pi0_state"],
        action_horizon=1,
        traject_horizon=(4, 5),
        ep_key="episode_idx",
        frame_key="frame_idx",
    )
    return Pi0LeRobotDataModule(
        repo_id=repo_id,
        dataset_config=cfg.to_dict(),
        debug_config=False,
        is_split_val=True,
        val_ratio=val_ratio,
        seed=seed,
        batch_size=batch_size,
        num_workers=num_workers,
        split_path=split_path,
        force_resplit=False,
    )


def make_policy(
    dm: Pi0LeRobotDataModule,
    stats_path: str,
    lr: float,
    weight_decay: float,
) -> RepViTHistorySingleStepPolicy:
    # Matches train_mshab_repvit_history_single_step.py exactly, except
    # that lr and weight_decay are supplied by the Optuna trial.
    return RepViTHistorySingleStepPolicy(
        extractor_type=EXTRACTOR_KEY,
        extractor_kwargs=dict(
            img_backbone_cfg={
                "head": (IMAGE_KEY, {"pretrained": False}, 128),
                "gripper": (IMAGE_KEY, {"pretrained": False}, 128),
            },
            img_out_feat={"head": None, "gripper": None},
            img_use_key=["head", "gripper"],
            vec_use_key=("qpos", "pi0_state"),
            vec_feat_encoder_type="res_mlp",
            vec_feat_encoder_kwargs=dict(
                hidden_dim=128, num_blocks=1, expansion=2, is_output_proj=False,
            ),
            vec_temproal_encoder_type=STATE_KEY,
            vec_temproal_encoder_kwargs=dict(
                out_dim=128, hidden_dim=128, initial_history_scale=0.1,
            ),
            vec_out_feat=None,
            traj_feat_encoder_type="res_mlp",
            traj_feat_encoder_kwargs=dict(
                hidden_dim=128, num_blocks=1, expansion=2, is_output_proj=False,
            ),
            traj_temproal_encoder_type=REFERENCE_KEY,
            traj_temproal_encoder_kwargs=dict(
                out_dim=128, hidden_dim=128, num_blocks=2, kernel_size=5,
            ),
            traj_out_feat=None,
        ),
        decoder_type="res_mlp",
        decoder_kwargs=dict(
            num_out_feats=256, hidden_dim=256, num_blocks=2, expansion=2,
            dropout_rate=0.1, is_output_proj=False,
        ),
        head_type="multi",
        head_kwargs=dict(head_decoder_config={
            "joint": ((0, 7), dict(mlp_layers=[128], dropout_rate=0.0)),
            "torso": ((7, 8), dict(mlp_layers=[128], dropout_rate=0.0)),
            "base": ((8, 10), dict(mlp_layers=[128], dropout_rate=0.0)),
        }),
        action_group={"joint": (0, 7), "torso": (7, 8), "base": (8, 10)},
        multi_head_weight={"joint": 1.0, "torso": 1.0, "base": 2.0},
        data_module=dm,
        stats_path=stats_path,
        image_method="none",
        vector_method="standard",
        action_method="minmax",
        traject_method="standard",
        loss_type="l1",
        key_metrics="MAE",
        lr=lr,
        lr_schedule_type="warmup_cos",
        lr_schedule_kwargs=dict(warmup_steps_ratio=0.05, min_lr_ratio=0.01),
        weight_decay=weight_decay,
    )


def export_results(study: optuna.Study, output_dir: Path) -> None:
    """Export even partial/pruned trials; SQLite remains source of truth."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "trial_results.csv"
    columns = [
        "trial", "state", "best_val_MAE", "lr", "weight_decay", "batch_size",
        "best_epoch", "best_global_step", "at_best_val_loss",
        "at_best_val_joint_MAE", "at_best_val_torso_MAE", "at_best_val_base_MAE",
    ]
    with results_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for trial in study.trials:
            row = {
                "trial": trial.number,
                "state": trial.state.name,
                "best_val_MAE": trial.value if trial.value is not None else trial.user_attrs.get("best_val_MAE", ""),
                "lr": trial.params.get("lr", ""),
                "weight_decay": trial.params.get("weight_decay", ""),
                "batch_size": trial.params.get("batch_size", ""),
            }
            for key in columns[6:]:
                row[key] = trial.user_attrs.get(key, "")
            writer.writerow(row)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        best = min(completed, key=lambda t: float(t.value))  # type: ignore[arg-type]
        payload: dict[str, Any] = {
            "trial": best.number,
            "best_val_MAE": float(best.value),
            "params": best.params,
            "best_epoch_zero_based": best.user_attrs.get("best_epoch"),
            "best_global_step": best.user_attrs.get("best_global_step"),
            "group_metrics_at_best_epoch": {
                key.removeprefix("at_best_"): value
                for key, value in best.user_attrs.items()
                if key.startswith("at_best_")
            },
            "study_name": study.study_name,
            "fixed_settings": study.user_attrs.get("fixed_settings"),
        }
        (output_dir / "best_params.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nBest trial #{best.number}: val_MAE={best.value:.6f}, params={best.params}")


def main(
    repo_id: str = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle_train",
    stats_path: str = "/root/vla_bc_policy_temporal/assets/mshab/tianyi_tidy_house.yaml",
    output_dir: str = "./output/mshab_repvit_optuna",
    study_name: str = "mshab_repvit_single_step_lr_wd_bs",
    n_trials: int = 15,
    max_epochs: int = 100,
    num_workers: int = 16,
    seed: int = 42,
    val_ratio: float = 0.05,
    split_path: str | None = None,
    is_fast_train: bool = False,
    enable_pruning: bool = False,
    save_best_checkpoint: bool = False,
) -> None:
    """Run n_trials ADDITIONAL trials; resumes an existing SQLite study."""
    if n_trials < 1 or max_epochs < 1 or num_workers < 0:
        raise ValueError("n_trials / max_epochs must be >=1, num_workers >=0")
    if not (0 < val_ratio < 1):
        raise ValueError("val_ratio must be in (0,1)")

    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "study.db"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_float32_matmul_precision("medium" if is_fast_train else "high")

    fixed_settings = {
        "repo_id": repo_id, "stats_path": str(Path(stats_path).expanduser().resolve()),
        "seed": seed, "val_ratio": val_ratio, "split_path": split_path,
        "max_epochs": max_epochs, "precision": "bf16-mixed" if is_fast_train else "32",
        "model": "repvit_m0_9, no pretraining, 5 state + 9 reference, 1 action",
        "loss": "l1, joint=1, torso=1, base=2",
        "lr_schedule": "warmup_cos, warmup=.05, min_ratio=.01",
        "search_space": "lr:[1e-4,5e-4] log, wd:[1e-5,1e-2] log, bs:{64,128}",
    }
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
    pruner: optuna.pruners.BasePruner
    if enable_pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=40, interval_steps=5
        )
    else:
        pruner = optuna.pruners.NopPruner()

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path.as_posix()}",
        engine_kwargs={"connect_args": {"timeout": 60}},
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )
    recorded_settings = study.user_attrs.get("fixed_settings")
    if recorded_settings is not None and recorded_settings != fixed_settings:
        raise ValueError(
            "Existing Optuna study has different fixed settings. "
            "Use a different --study-name or --output-dir.\n"
            f"Stored: {recorded_settings}\nNew: {fixed_settings}"
        )
    if recorded_settings is None:
        study.set_user_attr("fixed_settings", fixed_settings)
    # Make one baseline trial reproducible if this study starts empty.
    if not study.trials:
        study.enqueue_trial({"lr": 3e-4, "weight_decay": 5e-5, "batch_size": 128})

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [64, 128])

        # Same RNG seed and same persisted *episode-level* validation split.
        # Distinct batch sizes do entail distinct update counts per epoch.
        pl.seed_everything(seed, workers=True)
        dm = policy = trainer = None
        reporter = BestValidationReporter(trial)
        try:
            dm = make_datamodule(repo_id, bs, num_workers, seed, val_ratio, split_path)
            policy = make_policy(dm, stats_path, lr, wd)
            logger = TensorBoardLogger(
                save_dir=str(out_dir), name="trials", version=f"trial_{trial.number:03d}"
            )
            callbacks: list[Callback] = [reporter, LearningRateMonitor(logging_interval="step")]
            if save_best_checkpoint:
                callbacks.append(ModelCheckpoint(
                    dirpath=str(out_dir / "checkpoints" / f"trial_{trial.number:03d}"),
                    monitor="val_MAE", mode="min", save_top_k=1, save_last=False,
                    filename="best-{epoch:03d}-{val_MAE:.5f}",
                    auto_insert_metric_name=False,
                ))
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1,
                max_epochs=max_epochs,
                precision="bf16-mixed" if is_fast_train else "32",
                logger=logger,
                callbacks=callbacks,
                enable_checkpointing=save_best_checkpoint,
                enable_progress_bar=False,
                enable_model_summary=False,
                gradient_clip_val=2.0,
                gradient_clip_algorithm="norm",
                log_every_n_steps=50,
                num_sanity_val_steps=2,
                check_val_every_n_epoch=1,
            )
            trainer.fit(policy, datamodule=dm)
            if not math.isfinite(reporter.best_mae):
                raise RuntimeError("No validation MAE was logged during training")
            return reporter.best_mae
        finally:
            # Release previous trial's GPU and loader resources before next trial.
            if trainer is not None and trainer.logger is not None:
                trainer.logger.finalize("success" if math.isfinite(reporter.best_mae) else "failed")
            del policy, dm, trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def on_trial_finish(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        export_results(study, out_dir)
        print(
            f"Trial {trial.number} {trial.state.name}: "
            f"objective={trial.value}, params={trial.params}"
        )

    print(
        f"Optuna MS-HAB RepViT (from scratch), {n_trials} additional trials, "
        f"{max_epochs} epochs each.\n"
        f"Storage: {db_path}\nSearch: LR [1e-4,5e-4], WD [1e-5,1e-2], BS {{64,128}}\n"
        f"Pruning: {enable_pruning}, save per-trial best checkpoints: {save_best_checkpoint}"
    )
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=1, callbacks=[on_trial_finish], gc_after_trial=True)
    finally:
        export_results(study, out_dir)
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        print("No completed trials yet. Check errors / prune settings before selecting parameters.")


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
