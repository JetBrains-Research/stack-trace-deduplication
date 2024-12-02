import torch

from ea.sim.main.methods.neural.similarity import Similarity


class InnerProductSimilarity(Similarity):
    def __init__(self):
        super().__init__()
        self._min_value = torch.tensor(float("-inf"))

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        return (v1 * v2).sum(dim=1)

    @property
    def min_value(self) -> torch.Tensor:
        return self._min_value
