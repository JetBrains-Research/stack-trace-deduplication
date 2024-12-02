import torch

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation
from ea.sim.main.utils import device


class AvgAgg(Aggregation):
    def __init__(self, hidden_size: int, bidirectional: bool):
        super().__init__()
        self._bidirectional = bidirectional
        self._dim = 2 * hidden_size if bidirectional else hidden_size

    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, max_seq_len, D * hidden_size), D = 2 if bidir else 1.
        range_tensor = torch.arange(max(output_lens)).unsqueeze(0).expand(output.shape[0], -1).to(device)  # (batch_size, max_seq_len)
        expanded_seq_len = output_lens.unsqueeze(-1).expand_as(range_tensor).to(device)  # (batch_size, max_seq_len)
        mask: torch.Tensor = range_tensor < expanded_seq_len  # (batch_size, max_seq_len)
        denom = torch.sum(mask, -1, keepdim=True)
        avgs = torch.sum(output * mask.unsqueeze(-1), dim=1) / denom
        return avgs

    @property
    def dim(self) -> int:
        return self._dim
