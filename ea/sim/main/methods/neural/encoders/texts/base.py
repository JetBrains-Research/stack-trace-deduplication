from abc import abstractmethod, ABC
from typing import Iterable

import torch
from torch import nn

from ea.sim.main.methods.neural.encoders.objects import Item


class Encoder(nn.Module, ABC):
    def fit(self, items: Iterable[Item]) -> "Encoder":
        """
        Fitting additional classes for text encoder.
        :param items: Sequence of N items
        :return: TextEncoder
        """
        return self

    @abstractmethod
    def forward(self, items: list[Item]) -> torch.Tensor:
        """
        Encodes items into vectors.
        :param items: Sequence of N items
        :return: Tensor of size (N, dim)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def out_dim(self) -> int:
        """
        Returns item vector the dimension
        :return: Item vector dimension
        """
        raise NotImplementedError


