import torch

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation


class Attention(Aggregation):
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError

