from abc import ABC, abstractmethod

import torch
from torch import nn


class Similarity(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_value(self) -> torch.Tensor:
        raise NotImplementedError
