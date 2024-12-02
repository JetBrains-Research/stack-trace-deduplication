from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import nn

from ea.sim.main.methods.base import SimStackModel


class NeuralModel(nn.Module, SimStackModel, ABC):
    def __init__(self):
        super(NeuralModel, self).__init__()

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'NeuralModel':
        return self

    @abstractmethod
    def save(self, folder: Path):
        raise NotImplementedError

    @abstractmethod
    def load(self, folder: Path, *, device: torch.device | str):
        raise NotImplementedError

    def train(self, mode: bool = True):
        super().train(mode)

    def opt_params(self) -> list[torch.Tensor]:
        return self.agg.opt_params() + self.classifier.opt_params()
