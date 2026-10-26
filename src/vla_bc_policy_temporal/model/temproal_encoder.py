from typing import Any, Literal, Mapping, Optional
import warnings

import torch
import torch.nn as nn
from vla_bc_policy_temporal.model.mlp import MlpType, get_mlp
from vla_bc_policy_temporal.model.utility import NormType, ActivateFnType, ActivateFnDict

class MlpTemproalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        out_dim: int = 128,
        seq_len: int = 9,

        mlp_type: MlpType = "mlp",
        mlp_kwargs: Mapping[str, Any] = {},

        # 关注位置, 暂时没有相关实现
        extract_idx: int = -1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.seq_len = seq_len

        self.mlp_kwargs = dict(mlp_kwargs)
        self.mlp_kwargs["num_in_feats"] = self.input_dim * self.seq_len
        self.mlp_kwargs["num_out_feats"] = self.out_dim

        self.decoder = get_mlp(mlp_type)(**self.mlp_kwargs)

    def forward(self, X: torch.Tensor):
        # x: (B, T, F)
        B, T, F = X.shape

        assert T == self.seq_len
        assert F == self.input_dim

        return self.decoder(X.flatten(1))

###

class GRUTemproalEncoder(nn.Module):
    def __init__(
        self,

        input_dim: int = 128,
        out_dim: int = 128,
        seq_len: int = 9,

        # 是否将输入特征与经过时序处理的特征融合
        is_fuse_input: bool = True,

        # GRU 隐藏层大小
        hidden_dim: int = 128,
        # 关注位置, 默认取最后一帧作为输出, 如果取其他帧特征将使用双向 GRU
        extract_idx: int = -1,
        # 取 1 层即可, 优先通过增大 hidden dim 提升能力, 不建议取超过 2
        num_layers: int = 1,
    ) -> None:
        '''
        可以专注提取任意长度序列在感兴趣位置上的特征
        当 T 较长、感兴趣位置在末尾时使用
        '''
        
        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.seq_len = seq_len
        self.extract_idx = extract_idx

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.bidirectional = (extract_idx != -1)
        self.temporal = nn.GRU(
            input_size = hidden_dim,
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,

            bidirectional = self.bidirectional
        )

        self.is_fuse_input = is_fuse_input
        temporal_width = hidden_dim * (2 if self.bidirectional else 1)
        self.output_proj = nn.Linear(temporal_width + (hidden_dim if is_fuse_input else 0), out_dim)

    def forward(self, X: torch.Tensor):
        # x: (B, T, F)
        B, T, F = X.shape

        assert T == self.seq_len
        assert F == self.input_dim

        # Linear 仅对最后一层生效, 保留 (B, T, H)
        z = self.input_proj(X)

        _, h_n = self.temporal(z)
        # (B, H)
        extract_history = h_n[self.extract_idx]

        if self.is_fuse_input:
            raw = z[:, self.extract_idx]
            return self.output_proj(torch.cat(
                [extract_history, raw], dim = 1
            ))
        else:
            return self.output_proj(extract_history)

###

class ChannelLayerNorm1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor):
        # (B, C, T) -> (B, T, C) -> (B, C, T)
        return self.norm(
            x.transpose(1, 2)
        ).transpose(1, 2)

class ResidualConvBlock(nn.Module):
    def __init__(
        self, 
        channels: int,
        kernel_size: int = 3,
        is_norm: bool = True,
        is_res_connect: bool = True
    ):
        super().__init__()

        self.is_res_connect = is_res_connect

        self.norm1 = ChannelLayerNorm1d(channels) if is_norm else nn.Identity()
        self.norm2 = ChannelLayerNorm1d(channels) if is_norm else nn.Identity()

        self.act = nn.GELU()

        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size, padding = (kernel_size - 1) // 2
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size, padding = (kernel_size - 1) // 2
        )

    def forward(self, x: torch.Tensor):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.conv2(self.act(self.norm2(y)))

        if self.is_res_connect:
            return x + y
        else:
            return y

class CNNTemproalEncoder(nn.Module):
    def __init__(
        self,

        input_dim: int = 128,
        out_dim: int = 128,
        seq_len: int = 9,

        # 序列特征维度大小
        hidden_dim: int = 128,
        # 1D 卷积核大小 
        kernel_size: int = 3,
        # 卷积块数包含两个卷积层, 感受野大小为 1 + L * (K − 1) 应该与序列长度接近
        num_blocks: int = 2,
        # 卷积层中是否使用 LayerNorm
        is_norm: bool = True,
        # 卷积层中是否使用残差连接
        is_res_connect: bool = True,

        # 是否将输入特征与经过时序处理的特征融合
        is_fuse_input: bool = True,
        # 关注位置, 仅当 is_fuse_input 为 True 时有效
        extract_idx: int = -1,

    ) -> None:
        '''
        提取整个序列宏观特征

        需要增加层数解决感受野不足的问题
        需要后续与感兴趣点原始特征拼接保留局部特征
        
        当 T 较短且时间间隔一致时使用
        '''

        super().__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.seq_len = seq_len

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        backbone = [
            ResidualConvBlock(hidden_dim, kernel_size, is_norm, is_res_connect)
            for _ in range(num_blocks)
        ]
        backbone += [
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        ]
        self.backbone = nn.Sequential(*backbone)

        self.extract_idx = extract_idx
        self.is_fuse_input = is_fuse_input
        if self.is_fuse_input:
            self.output_proj = nn.Linear(
                2 * hidden_dim, out_dim
            )
        else:
            self.output_proj = nn.Linear(
                hidden_dim, out_dim
            )

    def forward(self, X: torch.Tensor):
        # x: (B, T, F)
        B, T, F = X.shape

        assert T == self.seq_len
        assert F == self.input_dim

        # Linear 仅对最后一层生效, 保留 (B, T, H)
        z = self.input_proj(X)
        global_feat = self.backbone(torch.transpose(z, 1, 2))

        if self.is_fuse_input:
            raw = z[:, self.extract_idx]
            return self.output_proj(torch.cat(
                [global_feat, raw], dim = 1
            ))
        else:
            return self.output_proj(global_feat)

TemproalEncoderType = Literal["mlp", "gru", "cnn"]
TemproalEncoderDict = {
    "mlp": MlpTemproalEncoder,
    "gru": GRUTemproalEncoder,
    "cnn": CNNTemproalEncoder,
}
def get_temproal_encoder(encoder: TemproalEncoderType):
    res = TemproalEncoderDict.get(encoder, None)
    if res is None:
        raise ValueError(f"Unknown encoder type {encoder}, available in {list(TemproalEncoderDict.keys())}")
    return TemproalEncoderDict[encoder]
