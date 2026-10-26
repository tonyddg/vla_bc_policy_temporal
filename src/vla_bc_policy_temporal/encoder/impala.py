from torch import nn
import torch

class ConvAutoPad(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = (kernel_size - 1) // 2,
            stride = stride,
        )
    
    def forward(self, X: torch.Tensor):
        return self.conv(X)

def make_gn(c: int, is_use_gn: bool = True) -> nn.Module:

    if not is_use_gn:
        return nn.Identity()

    if c <= 32:
        g = 4
    elif c <= 128:
        g = 8
    else:
        g = 16

    while c % g != 0:
        g //= 2
    return nn.GroupNorm(g, c)

class ImpalaResiduleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        is_use_gn: bool = False,
    ) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.ReLU(True),
            ConvAutoPad(
                in_channels,
                in_channels,
                kernel_size = 3
            ),
            make_gn(in_channels, is_use_gn),
            nn.ReLU(True),
            ConvAutoPad(
                in_channels,
                in_channels,
                kernel_size = 3
            ),
            make_gn(in_channels, is_use_gn),
        )

    def forward(self, X: torch.Tensor):
        return torch.add(self.main(X), X)

class ImpalaStage(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        # 仅使用单个残差层
        is_single_residule: bool = False,
        is_use_gn: bool = False,
    ) -> None:
        super().__init__()

        if is_use_gn:
            stage_model_list = [
                ConvAutoPad(
                    in_channels = in_channel,
                    out_channels = out_channel,
                    kernel_size = 3
                ),
                make_gn(out_channel, is_use_gn),
                nn.ReLU(),
                nn.MaxPool2d(
                    3, stride = 2, padding = 1
                ),
            ]
        else:
            stage_model_list = [
                ConvAutoPad(
                    in_channels = in_channel,
                    out_channels = out_channel,
                    kernel_size = 3
                ),
                nn.MaxPool2d(
                    3, stride = 2, padding = 1
                ),
            ]

        stage_model_list.append(
            ImpalaResiduleBlock(out_channel, is_use_gn)
        )
        if not is_single_residule:
            stage_model_list.append(
                ImpalaResiduleBlock(out_channel, is_use_gn)
            )

        self.stage = nn.Sequential(*stage_model_list)
    
    def forward(self, X: torch.Tensor):
        return self.stage(X)

class ImpalaEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: list[int] = [16, 32, 32],
        out_feat: int = 256,
        # 仅使用单个残差层
        is_single_residule: bool = False,
        is_use_gn: bool = False,
    ) -> None:
        super().__init__()

        backbone_model_list = []
        cnn_channels = [in_channels] + out_channels

        for idx in range(1, len(cnn_channels)):
            backbone_model_list.append(ImpalaStage(
                cnn_channels[idx - 1], cnn_channels[idx], is_single_residule = is_single_residule, is_use_gn = is_use_gn
            ))
        self.backbone = nn.Sequential(*backbone_model_list)

        self.bottelneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.ReLU(True),
            nn.Flatten(),

            nn.Linear(out_channels[-1], out_feat),
            nn.ReLU(True),
        )

    def forward(self, X: torch.Tensor):
        feat = self.backbone(X)
        return self.bottelneck(feat)

# if __name__ == "__main__":
    
#     from model_garage.cls.lit_module import ClsLitModule

#     model = ClsLitModule(
#         ImpalaEncoder, dict(),
#         input_size = (3, 96, 96),
#         decoder_kwargs = dict(
#             num_classes = 10
#         )
#     )