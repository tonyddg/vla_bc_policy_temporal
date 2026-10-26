from typing import Literal, Optional, Sequence

import torch
from torch import nn

NormType = Literal["none", "batch", "layer"]
ActivateFnType = Literal["relu", "silu", "gelu"]
ActivateFnDict = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU
}

LossFnType = Literal["mse", "l1", "smooth_l1"]
LossFnDict = {
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss
}

def get_model_output_shape(
    model: nn.Module,
    model_input: torch.Tensor,
    use_eval: bool = True,
):
    """
    给定模型和输入，通过一次 forward 推理获得模型输出形状。

    Args:
        model: PyTorch 模型对象。
        model_input: 模型输入
        use_eval: 是否临时切换到 eval 模式，默认 True。

    Returns:
        output_shape: 模型输出的 shape。
    """

    was_training = model.training

    if use_eval:
        model.eval()

    with torch.no_grad():
        if isinstance(model_input, dict):
            output = model(**model_input)

        elif isinstance(model_input, (tuple, list)):
            output = model(*model_input)

        else:
            output = model(model_input)

    if use_eval and was_training:
        model.train()

    return tuple(output.shape)[1:]

@torch.no_grad()
def regression_metrics(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    groups: Optional[dict[str, tuple[int, int]]] = None,
    low: float = -1.0,
    high: float = 1.0,
    thresholds: Sequence[float] = (0.05, 0.10, 0.20), 
    quantiles: Sequence[float] = (0.90, 0.95, 0.99),
):
    """
    y_pred: torch.Tensor, shape [B] / [B, 1] / [B, D] / ...
    y_true: torch.Tensor, same shape as y_pred
    """

    y_pred = y_pred.float()
    y_true = y_true.float()

    # 展平成一维，统计整体误差
    pred = y_pred.reshape(-1)
    true = y_true.reshape(-1)

    error = pred - true
    abs_error = torch.abs(error)

    mse = torch.mean(error ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(abs_error)
    medae = torch.median(abs_error)

    pred_dim = y_pred.reshape(y_pred.shape[0], -1)
    true_dim = y_true.reshape(y_true.shape[0], -1)
    num_dims = pred_dim.shape[1]

    metrics = {
        "MAE": mae.item(),
        "RMSE": rmse.item(),
        "MedAE": medae.item()
    }

    # 阈值准确率
    for t in thresholds:
        metrics[f"Acc@{t}"] = torch.mean((abs_error < t).float()).item()
    # 全样本阈值准确率
    sample_abs_error = torch.abs(pred_dim - true_dim)
    for t in thresholds:
        sample_correct = torch.all(sample_abs_error < t, dim=1)
        metrics[f"SampleAcc@{t}"] = sample_correct.float().mean().item()
    # 长尾误差特性
    flat_abs_err = abs_error.reshape(-1).float()
    for q in quantiles:
        metrics[f"ErrorP@{int(q * 100)}"] = torch.quantile(flat_abs_err, q).item()

    # 按给定索引区间统计 MSE / RMSE
    if groups is not None:
        for name, index_range in groups.items():
            a, b = index_range

            assert 0 <= a < b <= num_dims, \
                f"Invalid range for {name}: ({a}, {b}), but output dim is {num_dims}"

            group_pred = pred_dim[:, a:b]
            group_true = true_dim[:, a:b]

            group_error = group_pred - group_true

            group_mse = torch.mean(group_error ** 2)
            group_rmse = torch.sqrt(group_mse)
            group_mae = torch.mean(torch.abs(group_error))

            group_out_mask = (group_pred < low) | (group_pred > high)
            group_out_ratio = group_out_mask.float().mean()

            metrics[f"{name}_MAE"] = group_mae.item()
            metrics[f"{name}_RMSE"] = group_rmse.item()

            metrics[f"{name}_out_of_limit_ratio"] = group_out_ratio.item()
            # metrics[f"{name}_pred_max"] = group_pred.max().item()
            # metrics[f"{name}_pred_min"] = group_pred.min().item()

    return metrics

LRScheduleType = Literal[
    "none",             # 无学习率调度
    "warmup_cos",       # 预热余弦退火学习率调度
    "reduce"            # ReduceLROnPlateau 按指标调度学习率
]

import math
from torch.optim.lr_scheduler import LambdaLR

class WarmupCosineLR(LambdaLR):
    """
    Linear warmup + cosine decay scheduler.

    LR schedule:
        1. warmup: 0 -> base_lr
        2. cosine: base_lr -> min_lr_ratio * base_lr

    Args:
        optimizer:
            PyTorch optimizer.
        warmup_steps:
            Number of warmup optimizer steps.
        total_steps:
            Total number of optimizer steps.
        min_lr_ratio:
            Final LR ratio relative to base_lr.
            For example, 0.0 means decay to 0,
            0.1 means decay to 10% of base_lr.
        num_cycles:
            Number of cosine cycles. 0.5 means one half-cosine decay.
        last_epoch:
            PyTorch scheduler state. Keep default for normal use.
    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        warmup_steps_ratio: float = 0.05,

        min_lr_ratio: float = 0.0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")

        warmup_steps = int(warmup_steps_ratio * total_steps)
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps must be smaller than total_steps, "
                f"got warmup_steps={warmup_steps}, total_steps={total_steps}"
            )

        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(
                f"min_lr_ratio must be in [0, 1], got {min_lr_ratio}"
            )

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles

        super().__init__(
            optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=last_epoch,
        )

    def _lr_lambda(self, current_step: int) -> float:
        # 1. Linear warmup
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        # 2. Cosine decay
        progress = float(current_step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))

        cosine = 0.5 * (
            1.0 + math.cos(math.pi * 2.0 * self.num_cycles * progress)
        )

        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
