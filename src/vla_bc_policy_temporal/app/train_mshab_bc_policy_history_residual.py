import torch

from vla_bc_policy_temporal.dataset.pi0_lerobot_datamodule import Pi0LeRobotDataModule
from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import Pi0LeRobotDatasetConfig
from vla_bc_policy_temporal.dataset.camera_config import MshabCameraConfig

from vla_bc_policy_temporal.policy.history_residual_bc_policy import HistoryResidualBCPolicy
from vla_bc_policy_temporal.app.get_trainer import get_trainer

def main(
    repo_id: str = "trajectories_tidy_house_all_bc_with_qvel_state_rot_6d_action_axis_angle_train",
    stats_path: str = "/root/vla_bc_policy_temporal/assets/mshab/tianyi_tidy_house.yaml",
    exp_name: str = "history_residual_resnet_2221",
    # 启动低精度但更快的训练方式
    is_fast_train: bool = False,

    small_batch_lr: bool = False,
):
    if is_fast_train:
        torch.set_float32_matmul_precision('medium')
    else:
        torch.set_float32_matmul_precision('high')

    dm_config = Pi0LeRobotDatasetConfig(
        camera_config_dict = MshabCameraConfig,
        vec_obs_keys = ["qpos", "pi0_state"],
        observation_horizon = 5,
        padding_obs_zero_keys = ["pi0_state", ],

        action_horizon = 3,
        traject_horizon = (4, 5),

        ep_key = "episode_idx",
        frame_key = "frame_idx",
    )

    dm = Pi0LeRobotDataModule(
        repo_id = repo_id,
        dataset_config = dm_config.to_dict(),

        debug_config = False,
        is_split_val = True,
        val_ratio = 0.05,
        seed = 42,

        batch_size = 128 if not small_batch_lr else 64,
        num_workers = 64,
    )
    pm = HistoryResidualBCPolicy(
        extractor_type = "cur_img_history_residual_vec_traj",
        extractor_kwargs = dict(
            # Each camera: [image encoder, encoder kwargs, declared encoder output width].
            img_backbone_cfg = {
                "head": ("resnet", {"block_repeat": [2, 2, 2, 1]}, 512),
                "gripper": ("resnet", {"block_repeat": [2, 2, 2, 1]}, 512)
            },
            img_out_feat = {"head": 128, "gripper": 128},
            img_use_key = ["head", "gripper"],

            # Input dims and all horizon lengths are inferred from DataModule.
            vec_use_key = ("qpos", "pi0_state"),
            vec_feat_encoder_type = "res_mlp",
            vec_feat_encoder_kwargs = dict(
                hidden_dim = 128,
                num_blocks = 1,
                expansion = 2,
                is_output_proj = False
            ),
            vec_temproal_encoder_type = "history_residual_mlp",
            vec_temproal_encoder_kwargs = {"out_dim": 128, "hidden_dim": 128, "initial_history_scale": 0.1},
            vec_out_feat = None, # 不额外调整尺寸

            traj_feat_encoder_type = "res_mlp",
            traj_feat_encoder_kwargs = dict(
                hidden_dim = 128,
                num_blocks = 1,
                expansion = 2,
                is_output_proj = False
            ),
            traj_temproal_encoder_type = "cnn",
            traj_temproal_encoder_kwargs = {"out_dim": 128, "hidden_dim": 256, "num_blocks": 2},
            traj_out_feat = None # 不额外调整尺寸
        ),

        decoder_type = "res_mlp",
        decoder_kwargs = dict(
            num_out_feats = 512,
            hidden_dim = 512,
            num_blocks = 2,
            expansion = 2,
            dropout_rate = 0.1,
            is_output_proj = False,
        ),

        head_type = "multi",
        head_kwargs = dict(
            head_decoder_config = dict(
            joint = ((0, 7), dict(
                mlp_layers = [256],
                dropout_rate = 0.0
            )),
            torso = ((7, 8), dict(
                mlp_layers = [256],
                dropout_rate = 0.0
            )),
            base = ((8, 10), dict(
                mlp_layers = [256],
                dropout_rate = 0.0
            ))
        )),

        action_group = dict(
            joint = (0, 7),
            torso = (7, 8),
            base = (8, 10)
        ),
        # 增大 base 的权重
        multi_head_weight = dict(
            joint = 1.0, torso = 1.0, base = 2.0
        ),

        data_module = dm,
        stats_path = stats_path,
        image_method = "none",
        vector_method = "standard",
        action_method = "minmax",
        traject_method = "standard",

        loss_type = "l1",
        key_metrics = "MAE",
        lr = 3e-4 if not small_batch_lr else 1e-4,
        lr_schedule_type = "warmup_cos",
        lr_schedule_kwargs = dict(
            warmup_steps_ratio = 0.05,
            min_lr_ratio = 0.01
        ),
        weight_decay = 5e-5
    )
    
    trainer = get_trainer(
        exp_name, precision = "32" if not is_fast_train else "bf16-mixed", 
        max_epochs = 150, patience = None,
        gradient_clip_val = 2.0
    )
    trainer.fit(pm, datamodule = dm)

if __name__ == "__main__":

    import tyro
    tyro.cli(main)
