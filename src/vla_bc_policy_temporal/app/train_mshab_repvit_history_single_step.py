"""Train MS-HAB RepViT-M0.9 history-residual SINGLE-ACTION policy.

Usage (after copying the five new files into existing package directories):
  pip install 'timm>=1.0.15'
  python -m vla_bc_policy_temporal.app.train_mshab_repvit_history_single_step

For official timm ImageNet RepViT pretraining, enable `--pretrained` (network
access or cached weights required). For controlled architecture comparison,
keep pretrained=False and use the exact same dataset split and optimizer.
"""
import torch

from vla_bc_policy_temporal.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import Pi0LeRobotDatasetConfig
from vla_bc_policy_temporal.dataset.camera_config import MshabCameraConfig
from vla_bc_policy_temporal.policy.repvit_history_single_step_policy import (
    RepViTHistorySingleStepPolicy, IMAGE_KEY, STATE_KEY, REFERENCE_KEY, EXTRACTOR_KEY,
)
from vla_bc_policy_temporal.app.get_trainer import get_trainer


def main(
    repo_id: str = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle_train",
    stats_path: str = "/root/vla_bc_policy_temporal/assets/mshab/tianyi_tidy_house.yaml",
    exp_name: str = "mshab_repvit_m09_multiscale_history_single_step",
    pretrained: bool = False,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    max_epochs: int = 150,
    num_workers: int = 64,
    is_fast_train: bool = False,
) -> None:
    torch.set_float32_matmul_precision("medium" if is_fast_train else "high")
    config = Pi0LeRobotDatasetConfig(
        camera_config_dict=MshabCameraConfig,
        vec_obs_keys=["qpos", "pi0_state"],
        observation_horizon=5,
        padding_obs_zero_keys=["pi0_state"],
        action_horizon=1,             # Single current action, not a 3-step chunk
        traject_horizon=(4, 5),       # 9 points: t-4..t+4, current index=4
        ep_key="episode_idx",
        frame_key="frame_idx",
    )
    dm = Pi0LeRobotDataModule(
        repo_id=repo_id,
        dataset_config=config.to_dict(),
        debug_config=False,
        is_split_val=True,
        val_ratio=0.05,
        seed=42,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    policy = RepViTHistorySingleStepPolicy(
        extractor_type=EXTRACTOR_KEY,
        extractor_kwargs=dict(
            img_backbone_cfg={
                "head": (IMAGE_KEY, {"pretrained": pretrained}, 128),
                "gripper": (IMAGE_KEY, {"pretrained": pretrained}, 128),
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
        image_method="none", vector_method="standard",
        action_method="minmax", traject_method="standard",
        loss_type="l1", key_metrics="MAE",
        lr=lr, lr_schedule_type="warmup_cos",
        lr_schedule_kwargs=dict(warmup_steps_ratio=0.05, min_lr_ratio=0.01),
        weight_decay=weight_decay,
    )
    trainer = get_trainer(
        exp_name, precision="bf16-mixed" if is_fast_train else "32",
        max_epochs=max_epochs, patience=None,
        gradient_clip_val=2.0,
    )
    trainer.fit(policy, datamodule=dm)


if __name__ == "__main__":
    import tyro
    tyro.cli(main)
