from typing import Any, Dict, Literal, Mapping, Optional, Tuple

import torch
from torch import nn

from vla_bc_policy_temporal.encoder import get_image_encoder, ImageEncoderType
from vla_bc_policy_temporal.dataset.utility import VECTION_OBS_KEY, IMAGE_OBS_KEY, get_random_pi0_lerobot_batch

from vla_bc_policy_temporal.model.mlp import get_mlp, MlpType, MlpDecoder
from vla_bc_policy_temporal.model.utility import NormType, ActivateFnType, get_model_output_shape
from vla_bc_policy_temporal.model.temproal_encoder import get_temproal_encoder, TemproalEncoderType

def _filled(config: Mapping[str, Any] | None, **fixed) -> dict[str, Any]:
    """Add inferred dimensions without silently overwriting incorrect settings."""
    kwargs = dict(config or {})
    for key, value in fixed.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(f"{key}: configured {kwargs[key]!r}, dataset requires {value!r}")
        kwargs[key] = value
    return kwargs


# def _mlp(kind: str, params: Mapping[str, Any]) -> nn.Module:
#     if kind == "mlp":
#         return MlpDecoder(**params)
#     if kind == "res_mlp":
#         params = dict(params)
#         params.setdefault("hidden_dim", max(128, int(params["num_out_feats"])))
#         return ResMlpDecoder(**params)
#     raise ValueError(f"Invalid feature encoder: {kind}")

class ConcatFuseModule(nn.Module):
    def __init__(
        self,
        # 特征键名
        feat_name_list: list[str],
        # 调整配置, 与特征键名保持一致
        feat_config_list: list[Tuple[
            # 输入维度
            int, 
            # 合并前调整维度, 取 None 不会调整维度
            Optional[int], 
        ]],
        adjust_norm_type: NormType = "layer",
        adjust_activate_fn_type: ActivateFnType = "gelu",
    ) -> None:

        super().__init__()

        if len(feat_name_list) != len(set(feat_name_list)):
            raise ValueError("Feat name in feat_name_list is not unique")
        if len(feat_name_list) != len(feat_config_list):
            raise ValueError("Lenth of feat_name_list is not equal to feat_config_list")

        self.adjust_proj = nn.ModuleDict({
            feat_name : MlpDecoder(
                feat_config[0], feat_config[1], 
                is_output_proj = False,
                norm_type = adjust_norm_type,
                activate_fn_type = adjust_activate_fn_type
            ) if feat_config[1] is not None else nn.Identity()
            for feat_name, feat_config in zip(feat_name_list, feat_config_list)
        })

        self.out_feat = 0
        for feat_config in feat_config_list:
            if feat_config[1] is not None:
                self.out_feat += feat_config[1]
            else:
                self.out_feat += feat_config[0]

    def get_out_feat(self):
        return self.out_feat

    def forward(self, X: Mapping[str, torch.Tensor]):
        return torch.cat([
            proj(X[key])
            for key, proj in self.adjust_proj.items()
        ], dim = 1)

class CurImgTempVecTrajExtractor(nn.Module):
    def __init__(
        self,

        # 各个视觉观测通道数
        img_obs_dim: Dict[str, int],
        # 状态观测特征维度 (基于数据集视角)
        vec_obs_dim: Dict[str, int],
        # 状态观测特征维度
        traj_dim: int,
        # 状态时序长度
        vec_horizon: int,
        # 轨迹时序长度, 规则与 dataset 中相同
        traj_horizon: Tuple[int, int],

        # 使用的视觉观测名称
        img_use_key: list[str],
        # 各个视觉观测特征提取器
        img_backbone_cfg: Dict[str, Tuple[
            ImageEncoderType, # 特征提取网络类 
            Dict, # 特征提取网络参数 
            int, # 特征提取器输出的图像特征维度
        ]],
        # 各个视觉观测合并前的调整维度, 取 None 不调整
        img_out_feat: Dict[str, Optional[int]],

        # 状态特征提取器类型
        vec_feat_encoder_type: MlpType = "mlp",
        # 状态特征提取器类型配置
        vec_feat_encoder_kwargs: dict[str, Any] = {},
        # 状态时序特征提取器类型
        vec_temproal_encoder_type: TemproalEncoderType = "mlp",
        # 状态时序特征提取器配置
        vec_temproal_encoder_kwargs: dict[str, Any] = {},
        # 状态特征合并前的调整维度, 取 None 不调整
        vec_out_feat: Optional[int] = None,
        # 使用的状态观测名称
        vec_use_key: list[str] = [],

        # 轨迹点特征提取器类型
        traj_feat_encoder_type: MlpType = "mlp",
        # 轨迹点特征提取器类型配置
        traj_feat_encoder_kwargs: dict[str, Any] = {},
        # 轨迹点时序特征提取器类型
        traj_temproal_encoder_type: TemproalEncoderType = "mlp",
        # 轨迹点时序特征提取器配置
        traj_temproal_encoder_kwargs: dict[str, Any] = {},
        # 轨迹点特征合并前的调整维度, 取 None 不调整
        traj_out_feat: Optional[int] = None,
        # 是否使用参考轨迹输入
        use_traj: bool = True,

        # 是否将时序观测与轨迹中的 padding 在推理时置为 0
        is_set_padding_to_zero: bool = False,
    ):
        super().__init__()
        if vec_horizon < 1 or len(traj_horizon) != 2 or traj_horizon[0] < 0 or traj_horizon[1] < 1:
            raise ValueError("Invalid observation or trajectory horizon")
        if traj_dim < 0:
            raise ValueError("traj_dim must be nonnegative")
        
        self.vec_horizon = int(vec_horizon)
        self.traj_horizon = tuple(traj_horizon)
        self.traj_seq_len = sum(traj_horizon)
        self.traj_dim = traj_dim
        self.img_obs_dim = dict(img_obs_dim)
        self.vec_obs_dim = dict(vec_obs_dim)
        self.img_use_key = tuple(img_use_key if img_use_key is not None and len(img_use_key) else img_obs_dim)
        self.vec_use_key = tuple(vec_use_key if vec_use_key is not None and len(vec_use_key) else vec_obs_dim)

        self.is_set_padding_to_zero = is_set_padding_to_zero
        
        if len(set(self.vec_use_key)) != len(self.vec_use_key):
            raise ValueError("Repeated vector observation keys")
        if set(self.vec_use_key) - set(vec_obs_dim):
            raise ValueError(f"Unknown vector observation keys: {set(self.vec_use_key)-set(vec_obs_dim)}")
        if set(img_backbone_cfg) != set(img_obs_dim):
            raise ValueError("img_backbone_cfg must define exactly the configured camera names")
        if img_out_feat and set(img_out_feat) - set(img_obs_dim):
            raise ValueError("Unknown camera in img_out_feat")

        feature_names: list[str] = []
        feature_dims: list[tuple[int, int | None]] = []
        self.image_encoders = nn.ModuleDict()
        self.image_feat_dim: dict[str, int] = {}
        for i, cam in enumerate(self.img_use_key):
            config = img_backbone_cfg[cam]
            if len(config) != 3:
                raise ValueError(f"Camera {cam}: expected (backbone_type, kwargs, backbone_output_dim)")
            encoder_type, encoder_kwargs, output_dim = config
            output_dim = int(output_dim)
            if output_dim <= 0 or img_obs_dim[cam] <= 0:
                raise ValueError(f"Camera {cam}: invalid channel/feature dimensions")
            self.image_encoders[str(i)] = get_image_encoder(encoder_type)(
                **_filled(encoder_kwargs, in_channels=img_obs_dim[cam])
            )
            self.image_feat_dim[cam] = output_dim
            feature_names.append(f"image:{cam}")
            feature_dims.append((output_dim, (img_out_feat or {}).get(cam)))

        self.use_vector = bool(self.vec_use_key)
        if self.use_vector:
            dim_in = sum(int(vec_obs_dim[name]) for name in self.vec_use_key)
            if dim_in <= 0:
                raise ValueError("Vector input dimension must be positive")
            vec_args = dict(vec_feat_encoder_kwargs or {})
            vec_feat_dim = int(vec_args.get("num_out_feats", 128))
            self.vec_feat_encoder = get_mlp(vec_feat_encoder_type)(
                **_filled(vec_args, num_in_feats=dim_in, num_out_feats=vec_feat_dim),
            )
            vec_temporal_args = dict(vec_temproal_encoder_kwargs or {})
            vec_temporal_out = int(vec_temporal_args.get("out_dim", vec_feat_dim))
            self.vec_temproal_encoder = get_temproal_encoder(vec_temproal_encoder_type)(
                **_filled(
                    vec_temporal_args, input_dim = vec_feat_dim,
                    out_dim = vec_temporal_out, seq_len = vec_horizon, 
                    extract_idx = -1 # 观测序列中, 最后一个观测最重要
                )
            )
            feature_names.append("vector")
            feature_dims.append((vec_temporal_out, vec_out_feat))

        self.use_traj = use_traj
        if self.use_traj:
            traj_args = dict(traj_feat_encoder_kwargs or {})
            traj_feat_dim = int(traj_args.get("num_out_feats", 128))
            self.traj_feat_encoder = get_mlp(traj_feat_encoder_type)(
                **_filled(traj_args, num_in_feats=traj_dim, num_out_feats=traj_feat_dim),
            )
            traj_temporal_args = dict(traj_temproal_encoder_kwargs or {})
            traj_temporal_out = int(traj_temporal_args.get("out_dim", traj_feat_dim))
            self.traj_temproal_encoder = get_temproal_encoder(traj_temproal_encoder_type)(
                **_filled(
                    traj_temporal_args, input_dim = traj_feat_dim,
                    out_dim = traj_temporal_out, seq_len = self.traj_seq_len, 
                    extract_idx = traj_horizon[0] # 参考轨迹中, 当前目标点最重要
                )
            )
            feature_names.append("traject")
            feature_dims.append((traj_temporal_out, traj_out_feat))

        # 保证不同模型中 feature_names 顺序保持不变
        feature_combind = [(name, dim) for name, dim in zip(feature_names, feature_dims)]
        feature_combind.sort(key = lambda elem: elem[0])
        feature_names_sorted = []
        feature_dims_sorted = []
        for name, dim in feature_combind:
            feature_names_sorted.append(name)
            feature_dims_sorted.append(dim)
        
        self.fusion = ConcatFuseModule(feature_names_sorted, feature_dims_sorted)
        self.out_feat = self.fusion.get_out_feat()

        # Compatible with Lightning's example-input mechanism, using a small
        # synthetic batch instead of decoding the actual LeRobot dataset.
        mock_obs, _, mock_traj, mock_masks = get_random_pi0_lerobot_batch(
            vec_obs_dim=vec_obs_dim, img_obs_dim=img_obs_dim,
            observation_horizon=vec_horizon, action_horizon=1,
            traject_horizon=self.traj_seq_len, batch_size=1,
            traject_dim=traj_dim if self.use_traj else None, image_size=(64, 64),
        )
        self.example_input_array = {"obs": mock_obs, "traject": mock_traj, "masks": mock_masks}

    def get_out_feat(self) -> int:
        return self.out_feat

    def forward(self, obs: Mapping[str, Mapping[str, torch.Tensor]],
                traject: torch.Tensor | None = None,
                masks: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks or {}
        feature: dict[str, torch.Tensor] = {}
        img_obs = obs[IMAGE_OBS_KEY]
        vec_obs = obs[VECTION_OBS_KEY]
        obs_mask = masks.get("obs_valid_mask")
        traj_mask = masks.get("traject_valid_mask")
        if obs_mask is not None and (obs_mask.ndim != 2 or obs_mask.shape[1] != self.vec_horizon):
            raise ValueError("obs_valid_mask must have shape [B, observation_horizon]")
        if obs_mask is not None and not obs_mask[:, -1].bool().all():
            raise ValueError("Current image/observation must be valid")

        for i, cam in enumerate(self.img_use_key):
            x = img_obs[cam]
            if x.ndim != 5 or x.shape[2] != self.img_obs_dim[cam]:
                # 启用 self.is_img_obs_cur_only 时, 仅有最新的图像
                raise ValueError(f"Image {cam} must be [B,{self.vec_horizon} or 1,{self.img_obs_dim[cam]},H,W]; got {tuple(x.shape)}")
            y = self.image_encoders[str(i)](x[:, -1].contiguous())
            if y.ndim != 2 or y.shape[1] != self.image_feat_dim[cam]:
                raise ValueError(f"Backbone for {cam} returned {tuple(y.shape)}, configured {self.image_feat_dim[cam]} features")
            feature[f"image:{cam}"] = y

        if self.use_vector:
            vectors = [vec_obs[name] for name in self.vec_use_key]
            batch = vectors[0].shape[0]
            for name, x in zip(self.vec_use_key, vectors):
                if x.shape != (batch, self.vec_horizon, self.vec_obs_dim[name]):
                    raise ValueError(f"Vector {name} got {tuple(x.shape)}, expected [{batch},{self.vec_horizon},{self.vec_obs_dim[name]}]")
            x = torch.cat(vectors, dim=-1)
            if obs_mask is not None and self.is_set_padding_to_zero:
                x = x * obs_mask.to(x.device, dtype=x.dtype).unsqueeze(-1)
            x = self.vec_feat_encoder(x)
            if obs_mask is not None and self.is_set_padding_to_zero:
                x = x * obs_mask.to(x.device, dtype=x.dtype).unsqueeze(-1)
            feature["vector"] = self.vec_temproal_encoder(x)

        if self.use_traj:
            if traject is None or traject.ndim != 3 or traject.shape[1:] != (self.traj_seq_len, self.traj_dim):
                raise ValueError(f"Trajectory must be [B,{self.traj_seq_len},{self.traj_dim}]")
            t = traject
            if traj_mask is not None and self.is_set_padding_to_zero:
                t = t * traj_mask.to(t.device, dtype=t.dtype).unsqueeze(-1)
            t = self.traj_feat_encoder(t)
            if traj_mask is not None and self.is_set_padding_to_zero:
                t = t * traj_mask.to(t.device, dtype=t.dtype).unsqueeze(-1)
            feature["traject"] = self.traj_temproal_encoder(t)
        return self.fusion(feature)


# Registry maintained here so future extractor types need no separate train.py.
ExtractorType = Literal["cur_img_temp_vec_traj", "img_cur_vec_temp_traj"]
ExtractorDict = {
    "cur_img_temp_vec_traj": CurImgTempVecTrajExtractor,
    "img_cur_vec_temp_traj": CurImgTempVecTrajExtractor,
}
# Kept as a spelling alias within the current project, not as an old model API.
ImgCurVecTempTrajExtractor = CurImgTempVecTrajExtractor
