import torch
from torch import nn

from ea.sim.main.methods.neural.similarity import Similarity
from ea.sim.main.utils import device


class CosineSimilarity(Similarity):
    def __init__(self):
        super().__init__()
        self._cos = nn.CosineSimilarity(dim=1)
        self._min_value = torch.tensor(-1.0).to(device)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        out = self._cos(v1, v2)
        return out

    @property
    def min_value(self) -> torch.Tensor:
        return self._min_value
