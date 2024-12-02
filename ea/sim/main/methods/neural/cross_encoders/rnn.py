from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from loguru import logger
import torch.nn.functional as F

from ea.sim.dev.scripts.training.reranker_losses import RerankerLoss, BCELossWithLogits
from ea.sim.main.methods.neural.encoders.modules.rnn import LSTMEmb, GRUEmb
from ea.sim.main.methods.neural.cross_encoders.base import CrossEncoder
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.tokens import BOWTokenEncoder
from ea.sim.main.methods.neural.encoders.texts.rnn import RNNTextEncoder
from ea.sim.main.methods.neural.encoders.tokens.rnn import RNNTokenEncoder
from ea.sim.main.utils import device


@dataclass
class LSTMCrossEncoderConfig:
    d_input: int = 50
    hidden_size: int = 100
    output_size: int = 200
    dropout: float = 0.0
    n_ctx: int = 200
    token_encoder: str = "bow"
    vocab_size: int = 10000
    loss: RerankerLoss = BCELossWithLogits()

    def __str__(self):
        return f"{self.d_input}_{self.hidden_size}_{self.output_size}_{self.dropout}_{self.token_encoder}_{self.loss.name}"


class LSTMCrossEncoder(CrossEncoder):
    name = 'cross_encoder_lstm'

    def __init__(self, config: LSTMCrossEncoderConfig):
        super().__init__()
        self._config = config

        if config.token_encoder == "bow":
            self._token_encoder = BOWTokenEncoder(config.vocab_size, config.d_input, config.dropout,
                                                  max_seq_len=config.n_ctx)
        elif config.token_encoder == "rnn":
            self._token_encoder = RNNTokenEncoder(config.vocab_size, config.d_input, config.dropout,
                                                  max_seq_len=config.n_ctx)

        self._had_same_item_emb = nn.Parameter(torch.randn(self._token_encoder.dim))

        self._rnn_module = LSTMEmb(
            input_size=self._token_encoder.dim,
            hidden_size=config.hidden_size,
            out_size=config.output_size,
            dropout=config.dropout,
            bidirectional=True
        )
        self._head = nn.Sequential(
            nn.Linear(config.output_size * 2, config.output_size),
            nn.ReLU(),
            nn.Linear(config.output_size, 1)
        )

    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        """
        return similarity scores for each pair of items.
        :param first: Sequence of N items
        :param second: Sequence of N items
        :return: Tensor of size (N, )
        """

        embs_first, mask_first = self._token_encoder([item.tokens for item in first])  # (N, seq_len1, d_input)
        mask_first = mask_first.all(dim=2)
        lens_first = torch.tensor([min(len(item), self._config.n_ctx) for item in first])
        embs_second, mask_second = self._token_encoder([item.tokens for item in second])  # (N, seq_len2, d_input)
        lens_second = torch.tensor([min(len(item), self._config.n_ctx) for item in second])
        mask_second = mask_second.all(dim=2)

        has_same_item_first = [
            [item1 in second[i].tokens for item1 in first[i].tokens[-self._config.n_ctx:]] +
            [False] * (embs_first.shape[1] - len(first[i].tokens[-self._config.n_ctx:]))
            for i in range(len(first))
        ]
        has_same_item_first = torch.tensor(has_same_item_first, dtype=torch.float32, device=device)  # (N, seq_len1)

        has_same_item_second = [
            [item2 in first[i].tokens for item2 in second[i].tokens[-self._config.n_ctx:]] +
            [False] * (embs_second.shape[1] - len(second[i].tokens[-self._config.n_ctx:]))
            for i in range(len(second))
        ]
        has_same_item_second = torch.tensor(has_same_item_second, dtype=torch.float32, device=device)  # (N, seq_len2)

        # add vector to embeddings if there is the same item in the opposite sequence
        embs_first = embs_first + has_same_item_first.unsqueeze(2) * self._had_same_item_emb.reshape(1, 1, -1)
        embs_second = embs_second + has_same_item_second.unsqueeze(2) * self._had_same_item_emb.reshape(1, 1, -1)

        rnn_emb_first = self._rnn_module(embs_first, lens_first)  # (N, output_size)
        rnn_emb_second = self._rnn_module(embs_second, lens_second)  # (N, output_size)

        return self._head(torch.cat([rnn_emb_first, rnn_emb_second], dim=1)).reshape(-1)
