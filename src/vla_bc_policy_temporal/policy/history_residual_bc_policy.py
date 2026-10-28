"""Separate policy class for the lightweight history-residual experiment.

Imports register *additional* extractor/encoder keys. Existing BCPolicy,
CurImgTempVecTrajExtractor and GRUTemproalEncoder remain untouched.
Use HistoryResidualBCPolicy.load_from_checkpoint(...) for checkpoints
trained with this class.
"""

from vla_bc_policy_temporal.model.extractor import ExtractorDict
from vla_bc_policy_temporal.model.temproal_encoder import TemproalEncoderDict
from vla_bc_policy_temporal.model.history_residual_encoder import HistoryResidualMLP
from vla_bc_policy_temporal.model.history_residual_extractor import (
    HistoryResidualCurImgVecTrajExtractor,
)
from vla_bc_policy_temporal.policy.bc_policy import BCPolicy


HISTORY_EXTRACTOR_TYPE = "cur_img_history_residual_vec_traj"
HISTORY_ENCODER_TYPE = "history_residual_mlp"

# Extend existing registries without changing any of their original entries.
TemproalEncoderDict[HISTORY_ENCODER_TYPE] = HistoryResidualMLP
ExtractorDict[HISTORY_EXTRACTOR_TYPE] = HistoryResidualCurImgVecTrajExtractor


class HistoryResidualBCPolicy(BCPolicy):
    """BCPolicy variant explicitly selecting the history-residual extractor."""

    def __init__(self, extractor_type: str = HISTORY_EXTRACTOR_TYPE, **kwargs):
        if extractor_type != HISTORY_EXTRACTOR_TYPE:
            raise ValueError(
                f"HistoryResidualBCPolicy requires extractor_type={HISTORY_EXTRACTOR_TYPE!r}"
            )
        super().__init__(extractor_type=extractor_type, **kwargs)
