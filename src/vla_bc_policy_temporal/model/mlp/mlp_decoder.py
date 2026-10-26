from typing import Optional

import numpy as np
import torch
from torch import nn

from vla_bc_policy_temporal.model.utility import NormType, ActivateFnType, ActivateFnDict

class MlpDecoder(nn.Module):
    def __init__(
        self,

        num_in_feats: int,
        num_out_feats: int,

        mlp_layers: Optional[list[int]] = None,
        dropout_rate: Optional[float] = None,
        norm_type: NormType = "layer",
        activate_fn_type: ActivateFnType = "gelu",

        is_init_weight: bool = True,

        is_output_proj: bool = True,
    ) -> None:
        '''__init__ MLP 解码器

        Args:
            num_in_feats (int): 输入特征
            num_out_feats (int): 输出特征
            mlp_layers (Optional[list[int]], optional): 隐藏层维度列表. Defaults to None.
            dropout_rate (Optional[float], optional): 额外 MLP 中的 dropout, 取 None 不使用. Defaults to None.
            norm_type (NormType, optional): 归一化层类型. Defaults to "none".
            activate_fn (ActivateFnType, optional): 额外 MLP 中的激活函数. Defaults to "gelu".
            is_init_weight (bool, optional): 是否初始化 Linear 权重. Defaults to True.

            is_output_proj (bool, optional): 输出层是否有额外 Linear 用于将 hidden feat 调整为 output feat，当编码器输出即结果时可设为 True. Defaults to True.
        '''
        super().__init__()

        backbone_list = []

        if mlp_layers is None:
            mlp_layers = [num_out_feats]
        else:
            mlp_layers = list(mlp_layers) + [num_out_feats]

        backbone_list = []
        last_mlp_feats = num_in_feats
        last_layer_idx = len(mlp_layers) - 1
        
        for layer_idx, mlp_feats in enumerate(mlp_layers):
            # Linear
            backbone_list.append(
                nn.Linear(
                    last_mlp_feats, mlp_feats
                )
            )

            if (layer_idx != last_layer_idx) or (not is_output_proj):
                # NormLayer
                if norm_type == "none":
                    pass
                elif norm_type == "batch":
                    backbone_list.append(
                        nn.BatchNorm1d(mlp_feats)
                    )
                elif norm_type == "layer":
                    backbone_list.append(
                        nn.LayerNorm(mlp_feats)
                    )
                else:
                    raise RuntimeError(f"Unknown norm type: {norm_type}")

                # Activation
                backbone_list.append(
                    ActivateFnDict[activate_fn_type]()
                )
                # Dropout
                if dropout_rate is not None:
                    backbone_list.append(
                        nn.Dropout(
                            dropout_rate
                        )
                    )

            last_mlp_feats = mlp_feats

        self.decoder = nn.Sequential(*backbone_list)
        if is_init_weight:
            self.init_weight()

    def forward(self, X: torch.Tensor):
        return self.decoder(X)

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)