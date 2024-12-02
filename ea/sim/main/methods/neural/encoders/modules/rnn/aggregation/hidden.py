import torch

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation


class HiddenStateAgg(Aggregation):
    def __init__(self, hidden_size: int, bidirectional: bool):
        super().__init__()
        self._bidirectional = bidirectional
        self._dim = 2 * hidden_size if bidirectional else hidden_size

    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        if not self._bidirectional:
            return h_n[0]

        return torch.cat((h_n[0], h_n[1]), dim=1)

    @property
    def dim(self) -> int:
        return self._dim
