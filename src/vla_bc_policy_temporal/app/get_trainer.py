from typing import Optional

import lightning.pytorch as pl
import optuna

def get_trainer(
    # model: ClsLitModule,
    # data_module: pl.LightningDataModule,
    name: str,
    max_epochs: int = 50,

    overfit_batches: float = 0.0,
    precision: Optional[str] = None,

    key_metrics_name: str = "MAE",
    patience: Optional[int] = 10,

    gradient_clip_val: Optional[float] = 1.0,

    optuna_trial: Optional[optuna.Trial] = None
):
    from pathlib import Path
    import vla_bc_policy_temporal

    from lightning.pytorch.callbacks import (ModelCheckpoint, EarlyStopping, RichProgressBar, LearningRateMonitor)
    from lightning.pytorch.loggers import (TensorBoardLogger)
    
    import dotenv
    import os
    dotenv.load_dotenv()

    monitor_metrics = f"val_{key_metrics_name}"

    BC_POLICY_CKPT_DIR = os.environ.get("BC_POLICY_TEMPORAL_CKPT_DIR", (Path(vla_bc_policy_temporal.__file__).parent / "../../output").as_posix())
    logger = TensorBoardLogger(
        BC_POLICY_CKPT_DIR, 
        name
    )

    cb_list = []

    file_name = f"best-epoch-{{epoch:02d}}-acc-{{{monitor_metrics}:.3f}}"
    cb_list.append(ModelCheckpoint(
        filename = file_name,
        monitor = monitor_metrics, mode = "min",
        save_last = True
    ))
    cb_list.append(RichProgressBar(
        leave = False
    ))
    cb_list.append(LearningRateMonitor(
        "step"
    ))
    if patience is not None:
        cb_list.append(EarlyStopping(
            monitor = monitor_metrics, mode = "min", patience = patience, verbose = True
        ))

    if optuna_trial is not None:
        from optuna_integration import PyTorchLightningPruningCallback
        cb_list.append(
            PyTorchLightningPruningCallback(optuna_trial, monitor = monitor_metrics)
        )

    trainer = pl.Trainer(
        max_epochs = max_epochs, 
        logger = logger,
        callbacks = cb_list,

        overfit_batches = overfit_batches,
        precision = precision, # type: ignore

        gradient_clip_val = gradient_clip_val,
        gradient_clip_algorithm = "norm",
    )
    return trainer
    # trainer.fit(model, data_module)
