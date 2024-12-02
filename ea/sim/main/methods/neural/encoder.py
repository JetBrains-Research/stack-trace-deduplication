from abc import abstractmethod, ABC
from pathlib import Path

import torch
from torch import nn


class Encoder(ABC, nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> "Encoder":
        return self

    @abstractmethod
    def forward(self, stack_id: int) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def encode(self, stack_ids: list[int]) -> torch.Tensor:
        vectors = [self(stack_id).view(1, -1) for stack_id in stack_ids]
        return torch.cat(vectors, dim=0)

    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def opt_params(self) -> list[torch.Tensor]:
        return []

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def save(self, folder: Path):
        raise NotImplementedError

    @abstractmethod
    def load(self, folder: Path, *, device: torch.device | str, load_weights: bool = True, ):
        raise NotImplementedError
