from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from vla_bc_policy_temporal.dataset.camera_config import CameraConfig, parse_raw_camera_config_dict, to_raw_camera_config_dict
from vla_bc_policy_temporal.dataset.utility import ACTION_SAMPLE_KEY, TRAJ_SAMPLE_KEY, VECTION_OBS_KEY, IMAGE_OBS_KEY, EP_KEY, FRAME_KEY
from vla_bc_policy_temporal.dataset.observation_builder import ObservationBuilder

def _stack_obs(obs_steps: list[dict[str, dict[str, torch.Tensor]]], is_img_obs_cur_only: bool = True):
    seq_len = len(obs_steps)
    return {
        IMAGE_OBS_KEY: {
            key: torch.stack(
                [
                    torch.as_tensor(step[IMAGE_OBS_KEY][key])
                    for idx, step in enumerate(obs_steps) if (not is_img_obs_cur_only or (idx == seq_len - 1))
                ],
                dim = 0,
            )
            for key in obs_steps[0][IMAGE_OBS_KEY]
        },
        VECTION_OBS_KEY: {
            key: torch.stack(
                [
                    torch.as_tensor(step[VECTION_OBS_KEY][key])
                    for step in obs_steps
                ],
                dim = 0,
            )
            for key in obs_steps[0][VECTION_OBS_KEY]
        },
    }

def get_dataset_dir(repo_id: str, root: Optional[Union[str, Path]] = None) -> Path:
    if root is None:
        try:
            from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        except ImportError:
            from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME # type: ignore
        root = HF_LEROBOT_HOME
    return Path(root) / repo_id

@dataclass
class Pi0LeRobotDatasetConfig:
    camera_config_dict: Mapping[str, CameraConfig]
    vec_obs_keys: Sequence[str]
    # 读取过去 T 个观测 [t-T+1, ..., t]
    observation_horizon: int = 1
    # 前端超出观测取 0 观测键值 (否则重复第一个有效观测, 默认重复最前一个有效观测)
    padding_obs_zero_keys: Optional[Sequence[str]] = None
    # 是否仅使用当前一帧的图像观测, 而不使用历史图像观测
    is_img_obs_cur_only: bool = True

    action_sample_key: str = ACTION_SAMPLE_KEY
    # 读取未来 T 个动作 [t, ..., t+T-1]
    action_horizon: int = 1
    # 末端超出动作取 0 mask (否则重复最后一个有效动作, 默认重复最后一个有效动作)
    padding_action_zero_mask: Optional[list[bool]] = None
    # 裁剪动作到 [-1, 1]
    clip_action: bool = True
    
    traject_sample_key: str = TRAJ_SAMPLE_KEY
    # 读取过去 T1 个轨迹点 [t-T1, ..., t-1]
    # 读取未来 T2 个轨迹点 [t, ..., t+T2]
    traject_horizon: tuple[int, int] = (0, 1)
    # 超出两端轨迹取 0 mask (否则重复最后一个有效轨迹中间点, 默认重复最后一个有效动作)
    padding_traject_zero_mask: Optional[list[bool]] = None

    # 兼容旧数据集使用 episode_idx
    ep_key: str = EP_KEY
    # 兼容旧数据集使用 frame_idx
    frame_key: str = FRAME_KEY

    def __post_init__(self):
        if type(self.observation_horizon) is not int or self.observation_horizon <= 0:
            raise ValueError("observation_horizon must be a positive integer")
        if type(self.action_horizon) is not int or self.action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer")
        if (type(self.traject_horizon[0]) is not int
                or self.traject_horizon[0] < 0):
            raise ValueError("traject_horizon[0] must be a nonnegative integer")
        if type(self.traject_horizon[1]) is not int or self.traject_horizon[1] <= 0:
            raise ValueError("traject_horizon must be a positive integer")

    @classmethod
    def from_dict_config(cls, dict_config: Mapping):
        '''
        用于 lightning 自动创建
        '''
        kwargs = dict(dict_config)
        raw_camera_config_dict = kwargs.get("camera_config_dict", None)
        if isinstance(raw_camera_config_dict, Mapping):
            camera_config_dict = parse_raw_camera_config_dict(raw_camera_config_dict)
        else:
            raise ValueError("no camera_config_dict")
        kwargs["camera_config_dict"] = camera_config_dict
        return cls(**kwargs)

    def to_dict(self):
        obj = asdict(self)
        obj["camera_config_dict"] = to_raw_camera_config_dict(self.camera_config_dict)
        return obj

    def get_data_feature_dim(
        self,
        repo_id: str, 
        root: Optional[Union[str, Path]] = None,
    ):
        '''
        从 Lerobot 元数据中获取一个 sample 的特征维度

        认为 detach.h5 中不包含 vector 类型的数据, 否则直接调用将抛出 KeyError, 建议改为通过加载 dataset 获取特征维度
        '''
        info_json_path = get_dataset_dir(repo_id, root) / "meta/info.json"

        with open(info_json_path) as f:
            dataset_info = dict(json.load(f))

        vec_obs_dim = {
            key: int(dataset_info["features"][key]["shape"][0])
            for key in self.vec_obs_keys
        }
        img_obs_dim = {
            key: val.image_channel
            for key, val in self.camera_config_dict.items()
        }

        output_dim = int(dataset_info["features"][self.action_sample_key]["shape"][0])
        traject_dim = int(dataset_info["features"][self.traject_sample_key]["shape"][0])

        return output_dim, traject_dim, vec_obs_dim, img_obs_dim

    def get_horizon(
        self
    ):
        return self.action_horizon, (self.traject_horizon[0] + self.traject_horizon[1]), self.observation_horizon 

class Pi0LeRobotDataset(Dataset):
    """
    idx -> 当前时刻 t

    observation_horizon: T1
    action_horizon: T2

    obs:    [t-T1+1, ..., t]
    action: [t, ..., t+T2-1]

    超出 episode 边界的部分重复边界帧。
    """

    def __init__(
        self,
        repo_id: str,
        config: Pi0LeRobotDatasetConfig,

        root: Optional[Union[str, Path]] = None,
        detach_h5_path: Optional[Union[str, Path]] = None,
    ):
        # 兼容新旧 LeRobot 导入路径
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset # type: ignore

        self.ds = LeRobotDataset(repo_id, root=root)

        self.observation_horizon = config.observation_horizon
        self.action_horizon = config.action_horizon
        self.is_img_obs_cur_only = config.is_img_obs_cur_only

        self.traject_sample_key = config.traject_sample_key
        self.action_sample_key = config.action_sample_key

        self.traject_horizon = config.traject_horizon

        self.ep_key = config.ep_key
        self.frame_key = config.frame_key

        self.clip_action = config.clip_action

        self.sample_to_obs = ObservationBuilder(
            camera_config_dict = config.camera_config_dict,
            vec_obs_keys = config.vec_obs_keys,
        )

        # ds.root 是 LeRobot 实际解析后的数据集目录
        self.dataset_dir_path = Path(self.ds.root)
        self.detach_h5_path = (
            Path(detach_h5_path)
            if detach_h5_path is not None
            else self.dataset_dir_path / "detach.h5"
        )

        self._h5 = None
        self._h5_pid = None

        # 两端动作与观测重复设置
        self.padding_action_zero_mask = config.padding_action_zero_mask
        self.padding_traject_zero_mask = config.padding_traject_zero_mask
        self.padding_obs_zero_keys = config.padding_obs_zero_keys

        # 只读取未来动作与 episode 索引，避免解码未来图像
        hf_dataset = self.ds.hf_dataset
        self._action_view = hf_dataset.select_columns(
            [self.ep_key, self.action_sample_key]
        )
        # 只读取轨迹与 episode 索引
        self._traj_view = hf_dataset.select_columns(
            [self.ep_key, self.traject_sample_key]
        )

    def __len__(self):
        return len(self.ds)

    def _episode_frame(self, sample: dict):
        episode_idx = int(
            torch.as_tensor(sample[self.ep_key]).item()
        )
        frame_idx = int(
            torch.as_tensor(sample[self.frame_key]).item()
        )

        return episode_idx, frame_idx

    def _get_h5(self):
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "For detach.h5 loading, please install h5py"
            ) from exc

        # 不同 DataLoader worker 独立打开 HDF5
        pid = os.getpid()

        if self._h5 is not None and self._h5_pid != pid:
            self._h5.close()
            self._h5 = None

        if self._h5 is None:
            self._h5 = h5py.File(
                self.detach_h5_path,
                "r",
                swmr=True,
            )
            self._h5_pid = pid

        return self._h5

    def __getstate__(self):
        """避免多进程 pickle 时序列化 HDF5 句柄。"""
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def _load_detach_sample(self, sample: dict):
        """按 episode/frame 读取单帧附加数据。"""
        episode_idx, frame_idx = self._episode_frame(sample)

        episode_group = self._get_h5()[
            f"episode_{episode_idx:06d}"
        ]

        for key, dataset in episode_group.items(): # type: ignore
            sample[key] = torch.as_tensor(
                dataset[frame_idx]
            )

        return sample

    def _get_actions(self, idx: int, episode_idx: int):
        """读取未来 T2 个动作，不允许跨 episode。"""
        valid_actions = []

        # 优先从 Parquet 中仅查询动作列
        end = min(idx + self.action_horizon, len(self))
        rows = self._action_view[idx:end]

        for ep, action in zip(
            rows[self.ep_key],
            rows[self.action_sample_key],
        ):
            if int(torch.as_tensor(ep).item()) != episode_idx:
                break

            valid_actions.append(
                torch.as_tensor(action)
            )

        if not valid_actions:
            raise RuntimeError(
                "No valid action at current dataset index"
            )

        n_valid = len(valid_actions)

        # True 表示真实动作，False 表示填充动作
        action_mask = torch.arange(self.action_horizon) < n_valid

        # 构造重复动作
        padding_action = valid_actions[-1].clone()
        if self.padding_action_zero_mask is not None:
            if len(self.padding_action_zero_mask) != padding_action.numel():
                raise ValueError("padding_action_zero_mask length must equal action dimension")
            padding_action[self.padding_action_zero_mask] = 0

        # 不足 T2 时重复最后一个有效动作
        actions = torch.stack(
            valid_actions
            + [padding_action] * (self.action_horizon - n_valid),
            dim=0,
        )

        if self.clip_action:
            actions = actions.clamp(-1, 1)

        return actions, action_mask

    def _get_trajects(self, idx: int, episode_idx: int):
        """
        读取过去 T1 个中间点与未来 T2 个中间点构成 Traject，不允许跨 episode。
        读取过去 T1 个轨迹点 [t-T1, ..., t-1]
        读取未来 T2 个轨迹点 [t, ..., t+T2-1]
        """
        
        valid_trajects = []
        valid_mask = torch.zeros(self.traject_horizon[1] + self.traject_horizon[0], dtype = torch.bool)
        num_head_invalid = 0
        num_tail_invalid = 0

        end = idx + self.traject_horizon[1]
        start = idx - self.traject_horizon[0]
        for cur_idx in range(start, end):

            if cur_idx > len(self) - 1:
                num_tail_invalid += 1
                continue
            elif cur_idx < 0:
                num_head_invalid += 1
                continue

            ep = int(torch.as_tensor(self._traj_view[cur_idx][self.ep_key]).item())
            if ep != episode_idx:
                # Episode IDs are not guaranteed to be numerically sorted.
                if cur_idx < idx:
                    num_head_invalid += 1
                else:
                    num_tail_invalid += 1
                continue
            
            traj = torch.as_tensor(self._traj_view[cur_idx][self.traject_sample_key])
            valid_mask[cur_idx - start] = True
            valid_trajects.append(traj)

        if not valid_trajects:
            raise RuntimeError(
                "No valid traject at current dataset index"
            )

        # 构造重复动作
        padding_tail_action = valid_trajects[-1].clone()
        padding_head_action = valid_trajects[0].clone()
        if self.padding_traject_zero_mask is not None:
            if len(self.padding_traject_zero_mask) != padding_tail_action.numel():
                raise ValueError("padding_traject_zero_mask length must equal trajectory dimension")
            padding_tail_action[self.padding_traject_zero_mask] = 0
            padding_head_action[self.padding_traject_zero_mask] = 0

        # 不足 T2 时重复最后一个有效动作
        trajects = torch.stack(
            [padding_head_action] * num_head_invalid + 
            valid_trajects
            + [padding_tail_action] * num_tail_invalid,
            dim=0,
        )

        return trajects, valid_mask

    def __getitem__(self, idx: int):
        idx = int(idx)

        if not (0 <= idx < len(self)):
            raise IndexError(idx)

        current_sample = self.ds[idx]
        episode_idx, frame_idx = self._episode_frame(
            current_sample
        )

        # frame_idx 从 episode 的第 0 帧开始
        episode_start = idx - frame_idx

        if episode_start < 0:
            raise ValueError("frame_index is incompatible with the dataset index")
        first_ep, first_frame = self._episode_frame(self.ds[episode_start])
        if first_ep != episode_idx or first_frame != 0:
            raise ValueError(
                "frame_index must be contiguous and zero-based within each episode; "
                "dataset indexing is incompatible with frame_index"
            )

        # -------- 历史观测 --------

        obs_cache = {}
        obs_steps = []
        obs_mask_values = []
        padding_obs = None

        # 如果存在重复观测则构造
        if idx - self.observation_horizon + 1 < episode_start:
            first_sample = dict(
                current_sample if idx == episode_start
                else self.ds[episode_start]
            )

            first_ep, _ = self._episode_frame(first_sample)
            if first_ep != episode_idx:
                raise RuntimeError(
                    "Observation window crossed episode boundary"
                )

            first_sample = self._load_detach_sample(first_sample)
            # 缓存真实第一帧
            obs_cache[episode_start] = (
                self.sample_to_obs.get_obs(first_sample)
            )

            # 构建 padding 观测
            padding_sample = dict(first_sample)
            if self.padding_obs_zero_keys is not None:

                for key in self.padding_obs_zero_keys:
                    if key not in padding_sample:
                        raise KeyError(
                            f"repeat_obs_zero_keys: missing {key!r}"
                        )
                    padding_sample[key] = torch.zeros_like(padding_sample[key])

            padding_obs = self.sample_to_obs.get_obs(padding_sample)

        # 从最早的历史帧遍历到当前帧
        for offset in range(self.observation_horizon - 1, -1, -1):
            requested_idx = idx - offset

            is_valid = requested_idx >= episode_start
            obs_mask_values.append(is_valid)

            # 超出开头时重复 episode 第一帧
            sample_idx = max(
                episode_start,
                requested_idx,
            )

            if not is_valid:
                obs_steps.append(padding_obs)
                continue

            # 有效 frame 或没有 repeat_obs 采用旧实现
            if sample_idx not in obs_cache:
                frame_sample = dict(
                    current_sample
                    if sample_idx == idx
                    else self.ds[sample_idx]
                )

                actual_episode, actual_frame = self._episode_frame(frame_sample)
                expected_frame = frame_idx - (idx - sample_idx)
                if actual_episode != episode_idx or actual_frame != expected_frame:
                    raise RuntimeError(
                        "Observation window crossed an episode boundary, "
                        "or frame_index is noncontiguous within this episode"
                    )

                frame_sample = self._load_detach_sample(
                    frame_sample
                )

                obs_cache[sample_idx] = (
                    self.sample_to_obs.get_obs(frame_sample)
                )

            obs_steps.append(obs_cache[sample_idx])

        # [T1, ...]
        obs = _stack_obs(obs_steps, self.is_img_obs_cur_only)

        # -------- 未来动作 --------

        actions, action_mask = self._get_actions(idx, episode_idx,)
        trajects, traject_mask = self._get_trajects(idx, episode_idx,)

        masks = {
            "obs_valid_mask": torch.tensor(
                obs_mask_values,
                dtype=torch.bool,
            ),
            "action_valid_mask": action_mask,
            "traject_valid_mask": traject_mask,
        }
        return obs, actions, trajects, masks

    def get_sample_size(self):
        """返回单步特征维度，而不是时间窗口长度。"""
        obs, action, traject = self[0][:3]

        output_dim = int(action.shape[-1])
        traject_dim = int(traject.shape[-1])

        vec_obs_dim = {
            key: int(torch.as_tensor(obs).shape[1])
            for key, obs in obs[VECTION_OBS_KEY].items()
        }

        img_obs_dim = {
            key: int(torch.as_tensor(obs).shape[1])
            for key, obs in obs[IMAGE_OBS_KEY].items()
        }

        return output_dim, traject_dim, vec_obs_dim, img_obs_dim

    def get_frame_episodes_by_idx(self, idx: int):
        return self._episode_frame(self.ds[idx])

    def get_episode_indices(self) -> dict[int, list[int]]:
        """
        Read episode IDs cheaply from HF column, with contiguous-block checks.
        获取 episode_index 对应 dataset idx 的映射
        """
        
        episode_to_indices: dict[int, list[int]] = {}
        closed: set[int] = set()
        previous: Optional[int] = None
        for idx, (raw_id, raw_frame) in enumerate(zip(
            self.ds.hf_dataset[self.ep_key], self.ds.hf_dataset[self.frame_key]
        )):
            episode = int(torch.as_tensor(raw_id).item())
            frame = int(torch.as_tensor(raw_frame).item())
            if episode != previous:
                if previous is not None:
                    closed.add(previous)
                if episode in closed:
                    raise ValueError(f"Episode {episode} is noncontiguous in the dataset")
                previous = episode
            indexes = episode_to_indices.setdefault(episode, [])
            if frame != len(indexes):
                raise ValueError(
                    f"Episode {episode} has nonzero or noncontiguous frame_index "
                    f"at dataset index {idx}: expected {len(indexes)}, got {frame}"
                )
            indexes.append(idx)
        if sum(map(len, episode_to_indices.values())) != len(self):
            raise ValueError("Episode indexing length does not match Dataset")
        return episode_to_indices
    
if __name__ == "__main__":
    # 测试代码

    from vla_bc_policy_temporal.dataset.camera_config import TianyiCameraConfig
    REPO_ID = "tianyi_test"
    SAMPLE_IDX = 5
    TEST_HORIZON = 9
    USE_CONFIG = TianyiCameraConfig
    DRAW_IMAGE_KEY = "head"
    
    print("#### Test Pi0LeRobotDataset Start ####")
    config = Pi0LeRobotDatasetConfig(
        camera_config_dict = USE_CONFIG,
        vec_obs_keys = ["qpos", ],
        observation_horizon = TEST_HORIZON,
        action_horizon = TEST_HORIZON,
        traject_horizon = (TEST_HORIZON, TEST_HORIZON)
    )
    ds = Pi0LeRobotDataset(
        REPO_ID, config = config
    )
    output_dim, traject_dim, vec_obs_dim, img_obs_dim = ds.get_sample_size()
    print(f"output_dim: {output_dim}, traject_dim: {traject_dim}, vec_obs_dim: {vec_obs_dim}, img_obs_dim: {img_obs_dim}")

    obs_sample, action_sample, traject_sample, mask_sample = ds[SAMPLE_IDX]

    print(
        f"action shape: {action_sample.shape}\n"
        f"traject shape: {traject_sample.shape}"
    )
    print(f"get_data_feature_dim {config.get_data_feature_dim(REPO_ID)}")

    def draw_sample_iamge(
        image_sample: torch.Tensor,
        figsize: tuple[int, int] = (6, 6),
    ):
        from matplotlib import pyplot as plt
        from matplotlib.axes import Axes
        import math

        windows_ratio = figsize[0] / figsize[1]
        num_axes = image_sample.shape[0]
        ncols = math.ceil(math.sqrt(num_axes * windows_ratio))
        nrows = math.ceil(num_axes / ncols)

        fig, axes_ = plt.subplots(nrows, ncols)
        axes: list[Axes] = list(axes_.flat)

        fig.set_size_inches(figsize)
        fig.set_layout_engine("compressed")

        for i, axe in enumerate(axes):
            axe.imshow(torch.permute(image_sample[i], [1, 2, 0]))
            axe.set_title(f"t: {i}")

        plt.savefig("output/tmp.png")

    print(obs_sample[IMAGE_OBS_KEY][DRAW_IMAGE_KEY].shape)
    draw_sample_iamge(obs_sample[IMAGE_OBS_KEY][DRAW_IMAGE_KEY])