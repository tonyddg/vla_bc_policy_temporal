import random
from typing import Literal, Mapping, Optional
import numpy as np
import torch

EP_KEY = "episode_index" # 兼容旧数据集使用 episode_idx
FRAME_KEY = "frame_index" # 兼容旧数据集使用 frame_idx

ACTION_SAMPLE_KEY = "policy_action"
TRAJ_SAMPLE_KEY = "pi0_actions_ref"
VECTION_OBS_KEY = "vector"
IMAGE_OBS_KEY = "image"

ROT_TYPE = Literal["axis_angle", "rot_6d", "quat", "rpy_euler"]
STATE_ROT_LENGTH_DICT = {
    "axis_angle": 3,
    "rot_6d": 6,
    "quat": 4,
    "rpy_euler": 3
}

def depth_to_pointcloud(
    depth: torch.Tensor,
    K: torch.Tensor,
    T_wc: torch.Tensor,
    znear: float = 0.0,
    zfar: float = np.inf,
) -> torch.Tensor:
    """
    将深度图转换为世界坐标系点云。

    Args:
        depth: (H, W, 1)，深度图，单位与相机位姿/内参一致
        K:     (3, 3)，相机内参矩阵
        T_wc:  (4, 4)，相机坐标系到世界坐标系的齐次变换矩阵
               即 X_w = T_wc @ X_c_h
        znear: 最小深度
        zfar: 最大深度

    Returns:
        points_world: (H, W, 3)，世界坐标系点云
                      无效深度对应点为 (0, 0, 0)
    """
    if depth.ndim != 3 or depth.shape[-1] != 1:
        raise ValueError(f"depth should have shape (H, W, 1), got {tuple(depth.shape)}")

    if K.shape != (3, 3):
        raise ValueError(f"K should have shape (3, 3), got {tuple(K.shape)}")

    if T_wc.shape != (4, 4):
        raise ValueError(f"T_wc should have shape (4, 4), got {tuple(T_wc.shape)}")

    H, W, _ = depth.shape
    device = depth.device
    dtype = depth.dtype

    K = K.to(device=device, dtype=dtype)
    T_wc = T_wc.to(device=device, dtype=dtype)

    # 提取深度: (H, W)
    z = depth[..., 0]

    # 无效深度掩码
    invalid_mask = (~torch.isfinite(z)) | (z <= znear) | (z >= zfar)
    valid_mask = ~invalid_mask

    # 像素坐标网格
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )  # v: (H, W), u: (H, W)

    # 内参
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # 反投影到相机坐标系
    x_cam = (u - cx) / fx * z
    y_cam = (v - cy) / fy * z
    z_cam = z

    # 无效点先置零，避免传播 NaN/Inf
    x_cam = torch.where(valid_mask, x_cam, torch.zeros_like(x_cam))
    y_cam = torch.where(valid_mask, y_cam, torch.zeros_like(y_cam))
    z_cam = torch.where(valid_mask, z_cam, torch.zeros_like(z_cam))

    ones = torch.ones_like(z_cam)

    # 转换为 sapien 标准
    # 原始相机坐标为 [x_cam, y_cam, z_cam]
    # sapien 标准这里使用 [z_cam, -x_cam, -y_cam]
    points_cam_h = torch.stack(
        [z_cam, -x_cam, -y_cam, ones],
        dim=-1,
    )  # (H, W, 4)

    # 展平后做矩阵乘法
    points_cam_h_flat = points_cam_h.reshape(H * W, 4)          # (N, 4)
    points_world_h = points_cam_h_flat @ T_wc.T                 # (N, 4)

    points_world = points_world_h[:, :3].reshape(H, W, 3)

    # 再次保证无效点严格为 (0, 0, 0)
    points_world = torch.where(
        valid_mask.unsqueeze(-1),
        points_world,
        torch.zeros_like(points_world),
    )

    return points_world

def get_random_pi0_lerobot_batch(
 
    vec_obs_dim: Mapping[str, int], 
    img_obs_dim: Mapping[str, int],

    action_horizon: int,
    traject_horizon: int,
    observation_horizon: int,
    
    batch_size: int,

    output_dim: Optional[int] = None, 
    traject_dim: Optional[int] = None,

    image_size: tuple[int, int] = (128, 128),
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """
    随机生成一个 mock batch
    """

    h, w = image_size

    obs = {}

    obs[VECTION_OBS_KEY] = {
        key: torch.randn(
        (batch_size, observation_horizon, vec_obs_dim[key]), device = device, dtype = dtype,
    )
        for key in vec_obs_dim
    }

    obs[IMAGE_OBS_KEY] = {
        key: torch.randn(
        (batch_size, observation_horizon, img_obs_dim[key], h, w), device = device, dtype = dtype,
    )
        for key in img_obs_dim
    }

    if output_dim is not None:
        action = torch.randn(
            (batch_size, action_horizon, output_dim), device = device, dtype = dtype,
        )
    else:
        action = None

    if traject_dim is not None:
        traject = torch.randn(
            (batch_size, traject_horizon, traject_dim), device = device, dtype = dtype,
        )
    else:
        traject = None

    masks = {
        "obs_valid_mask": torch.ones(batch_size, observation_horizon, dtype=torch.bool, device=device),
        "action_valid_mask": torch.ones(batch_size, action_horizon, dtype=torch.bool, device=device),
        "traject_valid_mask": torch.ones(batch_size, traject_horizon, dtype=torch.bool, device=device),
    }
    return obs, action, traject, masks

def random_split_indices(
    A: int,
    B: int,
    seed: Optional[int] = None,
) -> tuple[list[int], list[int]]:
    """
    从 [0, 1, ..., A-1] 中随机选择 B 个不重复的数。

    返回:
        selected: 随机选中的 B 个数，可作为测试集 indices
        remaining: 剩余的 A-B 个数，可作为训练集 indices

    参数:
        A: 总数量
        B: 要随机选择的数量
        seed: 随机种子，用于复现实验
    """

    if A < 0:
        raise ValueError("A must be non-negative")

    if B < 0 or B > A:
        raise ValueError("B must satisfy 0 <= B <= A")

    rng = random.Random(seed)

    all_indices = list(range(A))

    selected = rng.sample(all_indices, B)
    selected_set = set(selected)
    remaining = [idx for idx in all_indices if idx not in selected_set]
    
    return selected, remaining
