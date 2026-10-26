from vla_bc_policy_temporal.encoder.convnext import ConvNeXtEncoder
from vla_bc_policy_temporal.encoder.efficient_net import EfficientNetEncoder
from vla_bc_policy_temporal.encoder.impala import ImpalaEncoder
from vla_bc_policy_temporal.encoder.resnet import ResNetEncoder

from typing import Literal
ImageEncoderType = Literal["convnext", "efficient_net", "impala", "resnet"]
ImageEncoderDict = {
    "convnext": ConvNeXtEncoder,
    "efficient_net": EfficientNetEncoder,
    "impala": ImpalaEncoder,
    "resnet": ResNetEncoder
}

def get_image_encoder(encoder: ImageEncoderType):
    res = ImageEncoderDict.get(encoder, None)
    if res is None:
        raise ValueError(f"Unknown encoder type {encoder}, available in {list(ImageEncoderDict.keys())}")
    return ImageEncoderDict[encoder]
