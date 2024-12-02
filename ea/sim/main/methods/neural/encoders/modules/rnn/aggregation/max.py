import torch

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation
from ea.sim.main.utils import device


class MaxAgg(Aggregation):
    min_value = -1_000_000

    def __init__(self, hidden_size: int, bidirectional: bool):
        super().__init__()
        self._bidirectional = bidirectional
        self._dim = 2 * hidden_size if bidirectional else hidden_size

    def forward(self, output: torch.Tensor, output_lens: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        # output.shape = (batch_size, max_seq_len, D * hidden_size), D = 2 if bidir else 1.
        range_tensor = torch.arange(max(output_lens)).unsqueeze(0).expand(output.shape[0], -1).to(device)  # (batch_size, max_seq_len)
        expanded_seq_len = output_lens.unsqueeze(-1).expand_as(range_tensor).to(device)  # (batch_size, max_seq_len)
        mask = range_tensor < expanded_seq_len  # (batch_size, max_seq_len)
        # I don't want to make inplace tensor modification (replacing padding values to "-inf" as usual do.
        # Coping tensors can be time-consuming.
        # So, added very small value (min_value) to the padding to make it very small.
        min_value_mask = (~mask) * self.min_value
        maxs = torch.max(output + min_value_mask.unsqueeze(-1), dim=1)[0]
        return maxs

    @property
    def dim(self) -> int:
        return self._dim
