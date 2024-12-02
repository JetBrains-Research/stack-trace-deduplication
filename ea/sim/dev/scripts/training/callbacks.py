import typing as tp
from pathlib import Path

from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateMonitor


class CallbackArgs(tp.NamedTuple):
    checkpoint_dir: Path


class SaveStartAndFinalCheckpointCallback(Callback):
    start_checkpoint_name: str = "start_model.ckpt"
    end_checkpoint_name: str = "end_model.ckpt"

    def __init__(self, checkpoint_dir: Path):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_fit_start(self, trainer, pl_module) -> None:
        checkpoint_path = self._checkpoint_dir / SaveStartAndFinalCheckpointCallback.start_checkpoint_name
        trainer.save_checkpoint(checkpoint_path)

    def on_fit_end(self, trainer, pl_module) -> None:
        checkpoint_path = self._checkpoint_dir / SaveStartAndFinalCheckpointCallback.end_checkpoint_name
        trainer.save_checkpoint(checkpoint_path)


def callback_factory(args: CallbackArgs) -> list[Callback]:
    return [
        checkpoint_on_step_callback(args.checkpoint_dir),
        checkpoint_on_epoch_callback(args.checkpoint_dir),
        start_and_final_checkpoint_callback(args.checkpoint_dir),
        lr_monitor_callback()
    ]


def checkpoint_on_step_callback(checkpoint_dir: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-step-checkpoint",
        every_n_train_steps=1_000,
        save_top_k=-1,
    )


def checkpoint_on_epoch_callback(checkpoint_dir: Path) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{epoch}-{step}-epoch-checkpoint",
        every_n_epochs=1,
        save_top_k=-1,
    )


def start_and_final_checkpoint_callback(checkpoint_dir: Path) -> SaveStartAndFinalCheckpointCallback:
    return SaveStartAndFinalCheckpointCallback(
        checkpoint_dir=checkpoint_dir
    )


def lr_monitor_callback() -> LearningRateMonitor:
    return LearningRateMonitor(
        logging_interval="step"
    )
