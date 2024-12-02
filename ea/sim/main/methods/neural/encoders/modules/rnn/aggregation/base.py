from abc import ABC, abstractmethod

import torch
from torch import nn


class Aggregation(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError
