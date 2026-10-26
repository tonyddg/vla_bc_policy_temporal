import torch
from torch import nn

class ResidualConv(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int | None = None,
            stride: int = 1,
        ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.main_side = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                # BN 本身已经有可学习的 weight 和 bias，Conv 的 bias 基本会被 BN 抵消
                kernel_size = 3, stride = stride, padding = 1, bias = False
            ), 
            nn.BatchNorm2d(out_channels), nn.ReLU(True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size = 3, stride = 1, padding = 1, bias = False
            ), 
            nn.BatchNorm2d(out_channels)
        )

        # 当输入通道与主干通道不同时, 需要使用 1x1 卷积调整通道数
        if out_channels != in_channels or stride != 1:
            self.res_side = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size = 1, stride = stride, padding = 0, bias = False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.res_side = nn.Identity()

        self.add_activate = nn.ReLU(True)
    
    def forward(self, X):
        main_feat = self.main_side(X)
        res_feat = self.res_side(X)
        return self.add_activate(torch.add(main_feat, res_feat))

class ResNetStage(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            num_resconv: int,
            shrink_feat: bool = True
        ):
        '''
        * `shrink_feat` 是否采用特征大小减半的结构
        '''
        super().__init__()

        first_stride = 1
        if shrink_feat:
            first_stride = 2

        blks = []
        for i in range(num_resconv):
            # 每个 ResNet Stage 中, 由第一个 ResBlock 负责转换通道数与降低特征尺寸
            if i == 0:
                blks.append(ResidualConv(
                    in_channels, out_channels, first_stride
                ))
            # 后续 ResBlock 仅提取特征
            else:
                blks.append(ResidualConv(
                    out_channels, out_channels, 1
                ))
        self.model = nn.Sequential(*blks)
    
    def forward(self, X):
        return self.model(X)

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        # 取 [3, 4, 6, 3] 变为 ResNet-34
        block_repeat: list[int] = [2, 2, 2, 2],
        # 取 [32, 32, 64, 128, 256] 变为半宽形式的 ResNet
        feat_channels: list[int] = [64, 64, 128, 256, 512]
    ):
        super().__init__()

        head = nn.Sequential(
            nn.Conv2d(in_channels, feat_channels[0], 7, 2, 3),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        
        stage_list = []
        for i in range(4):
            stage_list.append(
                ResNetStage(
                    feat_channels[i], feat_channels[i + 1], block_repeat[i], 
                    False if i == 0 else True # 第一个 Stage 没有对特征尺寸减半
                )
            )
        # s2 = ResNetStage(64, 64, block_repeat[0], False)
        # s3 = ResNetStage(64, 128, block_repeat[1], True)
        # s4 = ResNetStage(128, 256, block_repeat[2], True)
        # s5 = ResNetStage(256, 512, block_repeat[3], True)

        dense = nn.Sequential(
            # 同样使用了全局均值池化将图像转为特征
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        stage_list.insert(0, head)
        stage_list.append(dense)
        self.model = nn.Sequential(*stage_list)
        # self.model = nn.Sequential(
        #     head, s2, s3, s4, s5, dense
        # )

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        return self.model(X)

# if __name__ == "__main__":
    
#     from model_garage.cls.lit_module import ClsLitModule
#     model = ClsLitModule(
#         ResNetEncoder, dict(),
#         input_size = (3, 32, 32),
#         decoder_kwargs = dict(
#             num_classes = 10
#         )
#     )