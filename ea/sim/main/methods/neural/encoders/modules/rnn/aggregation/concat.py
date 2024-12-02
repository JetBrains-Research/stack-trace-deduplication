import torch

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation


class ConcatAgg(Aggregation):
    def __init__(self, aggs: list[Aggregation]):
        super().__init__()
        self._aggs = aggs
        self._dim = sum(agg.dim for agg in self._aggs)

    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        outs = [agg(output, output_lens, h_n) for agg in self._aggs]
        return torch.cat(outs, dim=1)

    @property
    def dim(self) -> int:
        return self._dim
