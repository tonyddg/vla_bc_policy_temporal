from abc import ABC, abstractmethod
import math
from typing import Dict, Literal, Mapping, Optional, Sequence

import numpy as np
import torch
from dataclasses import dataclass, asdict

from vla_bc_policy_temporal.dataset.utility import depth_to_pointcloud

@dataclass
class CameraConfig(ABC):
    '''
    一个图片观测对应一个 CameraInfo
    '''
    
    process_type: str = ""
    # lerobot 读取为 CHW, detach dataset 读取为 HWC
    layout: Literal["HWC", "CHW"] = "HWC"
    image_channel: int = 0

    @abstractmethod
    def build_frame(
        self,
        sample: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        '''
        输入单帧 H, W, C / C, H, W, 输出 C, H, W
        '''
        raise NotImplementedError()

    def standard_frame(self, frame: torch.Tensor):
        # 图像最终处理, 将图像转为 C, H, W, 并且将像素置 0
        if frame.ndim == 2:
            # A single-channel depth image is unambiguous in either layout.
            frame = frame.unsqueeze(-1) if self.layout == "HWC" else frame.unsqueeze(0)
        if frame.ndim != 3:
            raise ValueError(f"Image frame must have 2 or 3 dimensions, got {tuple(frame.shape)}")

        if self.layout == "HWC":
            if frame.shape[-1] != self.image_channel:
                raise ValueError(f"Expected HWC depth with C={self.image_channel}, got {tuple(frame.shape)}")
            frame = frame.permute(2, 0, 1)
        elif frame.shape[0] != self.image_channel:
            raise ValueError(f"Expected CHW depth with C={self.image_channel}, got {tuple(frame.shape)}")

        # Invalid (including zero) depth should not become a strong near-field
        # activation. Use zero AFTER applying the nonlinear conversion.
        valid = torch.isfinite(frame) & (frame > 0)
        frame = torch.where(valid, frame, torch.zeros_like(frame))

        return frame

@dataclass
class DepthTanhCameraConfig(CameraConfig):

    depth_key: str = ""
    z_near: float = 0.01
    z_far: float = 2.0
    depth_scale: float = 1000.0

    def __post_init__(self):
        self.process_type = "depth_tanh"
        self.image_channel = 1

        if not self.depth_key:
            raise ValueError("depth_key must not be empty")
        if not all(math.isfinite(x) for x in (self.z_near, self.z_far, self.depth_scale)):
            raise ValueError("Depth camera scaling and clipping parameters must be finite")
        if not (self.depth_scale > 0 and self.z_far > self.z_near):
            raise ValueError("Require depth_scale > 0 and z_far > z_near")

    def build_frame(self, sample: Mapping[str, torch.Tensor]):
        if self.depth_key not in sample:
            raise KeyError(f"Depth camera input {self.depth_key!r} missing")

        frame = torch.as_tensor(sample[self.depth_key], dtype=torch.float32)
        frame = frame.type(torch.float32) / self.depth_scale
        
        frame = 1 - torch.tanh((frame - self.z_near) / (self.z_far - self.z_near))
        frame[frame < 0] = 0

        return self.standard_frame(frame)

    def get_image_channels(self) -> int:
        return 1

CameraProcessTypeMap = dict(
    depth_tanh = DepthTanhCameraConfig
)

def parse_raw_camera_config_dict(
    camera_config_dict: Mapping[str, Dict]
):
    result: dict[str, CameraConfig] = {}
    for name, raw in camera_config_dict.items():
        if isinstance(raw, CameraConfig):
            result[name] = raw
            continue
        if not isinstance(raw, Mapping):
            raise TypeError(f"Camera config {name!r} must be a mapping or CameraConfig")
        kwargs = dict(raw)

        process_type = kwargs.get("process_type", "")
        cls = CameraProcessTypeMap.get(process_type, None)
        if cls is None:
            raise ValueError(f"Unknown camera process_type {process_type!r}; supported {list(CameraProcessTypeMap)}")
        result[name] = cls(**kwargs)
    return result

def to_raw_camera_config_dict(
    camera_config_dict: Mapping[str, CameraConfig]
):
    return {name: asdict(config) for name, config in camera_config_dict.items()}

TianyiCameraConfig = dict(
    head = DepthTanhCameraConfig(
        layout = "HWC",
        depth_key = "head_depth",
        z_near = 0.15,
        z_far = 1.5,
        depth_scale = 1000.0
    )
)

MshabCameraConfig = dict(
    head = DepthTanhCameraConfig(
        layout = "HWC",
        depth_key = "head_depth",
        z_near = 0.01,
        z_far = 1.0,
        depth_scale = 1000.0
    ),
    gripper = DepthTanhCameraConfig(
        layout = "HWC",
        depth_key = "gripper_depth",
        z_near = 0.01,
        z_far = 1.0,
        depth_scale = 1000.0
    )
)

if __name__ == "__main__":
    raw = to_raw_camera_config_dict(TianyiCameraConfig)
    renew = parse_raw_camera_config_dict(raw)

    print(f"raw: {raw}")
    print(f"renew: {renew}")
