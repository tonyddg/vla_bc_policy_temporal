from typing import Literal, Optional, Sequence

from torch.nn import functional as F
from torch import nn
import torch

def drop_path(x, drop_prob: Optional[float] = None, training: bool = False):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    '''
    if drop_prob is None or drop_prob == 0.0 or not training: 
        return x
    
    keep_prob = 1 - drop_prob       # 保留的比率
    # 将单个 Batch 中随机几个样本置 0, 似乎不如 DropOut ?
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # 利用 floor 判断是否大于 1 实现二值化

    # 保证均值不变
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob: Optional[float] = None) -> None:
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).

    支持图片 channels_first 输入的 LayerNorm (原始 Pytorch 的 Layernorm 仅支持 channels_last)

    """

    def __init__(self, normalized_shape, eps: float = 1e-6, data_format: Literal["channels_last", "channels_first"] = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise ValueError(f"not support data format '{self.data_format}'")

class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        drop_rate: float = 0,
        # 可学习的全局特征缩放参数初始值
        layer_scale_init_value: float = 1e-6
    ) -> None:
        super().__init__()

        # 使用大卷积核做逐层卷积
        self.dwconv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size = 7, padding = 3, groups = in_channels
        )
        # 使用 permute 将特征移到最后做全连接, 代替 1x1 卷积, 且升维与降为完全在 1x1 卷积中完成
        self.pwconv1 = nn.Linear(in_channels, in_channels * 4)
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.norm = LayerNorm(in_channels, 1e-6, "channels_last")
        self.act = nn.GELU()
        # 可学习的全局特征缩放参数
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((in_channels, )), requires_grad = True
        ) if layer_scale_init_value > 0 else None
        # 随机丢弃网络自路径
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        shortcut = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
    
class DownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,

        kernel_size: int = 2,
        # 用于第一个 DownSample, 先卷积再标准化
        is_inverse: bool = False
    ) -> None:
        super().__init__()

        if is_inverse:
            self.down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size
                ),
                LayerNorm(
                    out_channels, 1e-6, data_format = "channels_first"
                )
            )
        else:
            self.down_sample = nn.Sequential(
                LayerNorm(
                    in_channels, 1e-6, data_format = "channels_first"
                ),
                nn.Conv2d(
                    in_channels, out_channels, kernel_size = kernel_size, stride = kernel_size
                )
            )
  
    def forward(self, x: torch.Tensor):
        return self.down_sample(x)

class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        # ConvNeXt Tiny 配置
        depths: Sequence[int] = (3, 3, 9, 3),
        stage_channels: Sequence[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ) -> None:
        super().__init__()

        # 构建每个stage中堆叠的block
        self.stages = nn.ModuleList()
        # 各个 stage 中的 drop_path_rate 按之前 block 数等差增大
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.num_stages = len(stage_channels)
        for i in range(self.num_stages):
   
            stage = []

            if i == 0:
                # 第一个下采样卷积
                stage.append(
                    DownsampleBlock(in_channels, stage_channels[i], 4, True)
                )
            else:
                # 后续下采样卷积
                stage.append(
                    DownsampleBlock(stage_channels[i - 1], stage_channels[i], 2, False)
                )

            for j in range(depths[i]):
                stage.append(
                    ConvNeXtBlock(
                        in_channels = stage_channels[i], 
                        drop_rate = dp_rates[cur + j], 
                        layer_scale_init_value = layer_scale_init_value
                    )
                )
            
            self.stages.append(nn.Sequential(*stage))
            # cur代表在当前Stage之前构建好了的block的个数
            cur += depths[i]

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(stage_channels[-1], eps = 1e-6)
        )
        # 设置初始权重
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std = 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor):
        for i in range(self.num_stages):
            x = self.stages[i](x)
        return self.head(x)

# if __name__ == "__main__":
    
#     from model_garage.cls.lit_module import ClsLitModule
#     model = ClsLitModule(
#         ConvNeXtEncoder, dict(),
#         input_size = (3, 64, 64),
#         decoder_kwargs = dict(
#             num_classes = 10
#         )
#     )