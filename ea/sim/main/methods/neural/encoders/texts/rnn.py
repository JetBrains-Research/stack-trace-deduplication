from typing import Iterable

from ea.sim.main.methods.neural.encoders.tokens.skip_gram_BOW import SkipGramBOW
import torch

from ea.sim.main.methods.neural.encoders.modules.rnn import LSTMEmb, GRUEmb
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.methods.neural.encoders.tokens import BOWTokenEncoder, RNNTokenEncoder


class RNNTextEncoder(Encoder):
    def __init__(
            self,
            vocab_size: int,
            token_emb_dim: int,
            hidden_size: int,
            out_size: int,
            rnn_type: str = "lstm",
            dropout: float = 0.0,
            bidirectional: bool = True,
            token_encoder: str = "rnn",
    ):
        super().__init__()
        if token_encoder == "rnn":
            self._token_encoder = RNNTokenEncoder(vocab_size, token_emb_dim, dropout)
        elif token_encoder == "bow":
            self._token_encoder = BOWTokenEncoder(vocab_size, token_emb_dim, dropout)
        elif token_encoder == "deep_crash":
           self._token_encoder = SkipGramBOW(token_emb_dim, dropout, window=1)
        else:
            raise ValueError(f"Unknown token encoder type: {token_encoder}")

        if rnn_type == "lstm":
            self._rnn_module = LSTMEmb(
                input_size=self._token_encoder.dim,
                hidden_size=hidden_size,
                out_size=out_size,
                dropout=dropout,
                bidirectional=bidirectional
            )
        elif rnn_type == "gru":
            self._rnn_module = GRUEmb(
                input_size=self._token_encoder.dim,
                hidden_size=hidden_size,
                out_size=out_size,
                dropout=dropout,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unknown rnn module type: {rnn_type}")

    def fit(self, items: Iterable[Item], total: int) -> "Encoder":
        self._token_encoder.fit((token for item in items for token in item.tokens))
        return self

    def forward(self, items: list[Item]) -> torch.Tensor:
        embeds, _ = self._token_encoder([[token for token in item.tokens] for item in items])
        lens = torch.tensor([len(item) for item in items])
        return self._rnn_module(embeds, lens)

    @property
    def out_dim(self) -> int:
        return self._rnn_module.dim
