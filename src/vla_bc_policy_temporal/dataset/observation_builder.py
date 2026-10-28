from typing import Sequence, Mapping

import torch

from vla_bc_policy_temporal.dataset.camera_config import CameraConfig
from vla_bc_policy_temporal.dataset.utility import VECTION_OBS_KEY, IMAGE_OBS_KEY

SampleType = Mapping[str, torch.Tensor]

class ObservationBuilder:
    def __init__(
        self,
        camera_config_dict: Mapping[str, CameraConfig],
        vec_obs_keys: Sequence[str], 
    ) -> None:

        if len(vec_obs_keys) != len(set(vec_obs_keys)):
            raise ValueError("vec_obs_keys contains duplicate feature keys")

        self.camera_config_dict = camera_config_dict
        self.vec_obs_keys = vec_obs_keys

    def get_obs(self, sample: SampleType, is_batched: bool = False) -> dict[str, dict[str, torch.Tensor]]:
        images = {name: camera.build_frame(sample, is_batched = is_batched)
                  for name, camera in self.camera_config_dict.items()}
        vectors: dict[str, torch.Tensor] = {}
        for name in self.vec_obs_keys:
            if name not in sample:
                raise KeyError(f"Vector observation {name!r} not found; available: {list(sample)}")
            if is_batched:
                vectors[name] = torch.as_tensor(sample[name], dtype = torch.float32).flatten(1)
            else:
                vectors[name] = torch.as_tensor(sample[name], dtype = torch.float32).flatten()
        return {IMAGE_OBS_KEY: images, VECTION_OBS_KEY: vectors}
