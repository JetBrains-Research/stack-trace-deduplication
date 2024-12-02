from abc import ABC, abstractmethod
from typing import Iterable, TypeAlias

import torch
from torch import nn

from ea.sim.main.preprocess.token import PostTokItem

IterItems: TypeAlias = Iterable[PostTokItem[int]]
ListItems: TypeAlias = list[PostTokItem[int]]
BatchListItems: TypeAlias = list[ListItems]


class TokenEncoder(nn.Module, ABC):
    """
    This class is responsible for encoding each token into a fixed-size vector, also known as an embedding. A token
    can either be atomic (e.g., a method name which is treated as a single token) or divided into multiple
    sub-tokens. Each sub-token is processed independently and the resulting vectors are subsequently combined.
    """

    def fit(self, items: IterItems) -> "TokenEncoder":
        """
        Fitting additional classes for token encoder.
        :param items: Sequence of N items
        :return: TokenEncoder
        """
        return self

    @abstractmethod
    def encode(self, items: ListItems) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def encode_batch(self, items: BatchListItems) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError

    def forward(self, items: ListItems | BatchListItems) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Encodes items or batch of items into vectors, batch processing is required for faster inference.
        :param items: Sequence of N items or batch
        :return: Tensor of size (N, dim)
        """
        if isinstance(items[0], PostTokItem):
            return self.encode(items)
        elif isinstance(items[0][0], PostTokItem):
            return self.encode_batch(items)
        else:
            raise ValueError(f"Inappropriate input type for 'items' argument: {type(items)}")

    @property
    @abstractmethod
    def dim(self) -> int:
        """
        Returns item vector the dimension
        :return: Item vector dimension
        """
        raise NotImplementedError
