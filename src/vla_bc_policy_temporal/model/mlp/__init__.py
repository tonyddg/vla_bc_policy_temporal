from vla_bc_policy_temporal.model.mlp.res_mlp_decoder import ResMlpDecoder
from vla_bc_policy_temporal.model.mlp.mlp_decoder import MlpDecoder

from typing import Literal
MlpType = Literal["mlp", "res_mlp"]
MlpDict = {
    "mlp": MlpDecoder,
    "res_mlp": ResMlpDecoder,
}

def get_mlp(encoder: MlpType):
    res = MlpDict.get(encoder, None)
    if res is None:
        raise ValueError(f"Unknown mlp type {encoder}, available in {list(MlpDict.keys())}")
    return MlpDict[encoder]
