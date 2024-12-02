from abc import ABC

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ea.sim.main.methods.neural.encoders.modules.rnn.aggregation import Aggregation, HiddenStateAgg, \
    ConcatAgg, MaxAgg, AvgAgg
from ea.sim.main.preprocess.id_coder import SpecialTokens

PAD_ID = SpecialTokens.PAD.id


class RNNEmb(nn.Module, ABC):
    def __init__(
            self,
            encoder: nn.LSTM | nn.GRU,
            agg: Aggregation,
            out_size: int,
            dropout: float = 0.0
    ):
        super().__init__()
        self._encoder = encoder
        self._agg = agg
        self._out_size = out_size
        self._mixture = nn.Linear(self._agg.dim, self._out_size)
        self._dropout = nn.Dropout(dropout)

    def forward(self, tensors: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """
        Computes the embeddings for a list of tensors (texts).
        :param tensors: Input of shape (n_texts, n_tokens, token_emb_dim)
        :param lens: Lengths of sequences
        :return: Embeddings of texts, shape (n_texts, text_emb_dim)
        """

        tensors = pack_padded_sequence(tensors, lens, batch_first=True, enforce_sorted=False)

        if isinstance(self._encoder, nn.LSTM):
            output, (h_n, _) = self._encoder(tensors)
        elif isinstance(self._encoder, nn.GRU):
            output, h_n = self._encoder(tensors)
        else:
            raise ValueError(f"Unknown type of encoder {type(self._encoder)}")

        output, output_lens = pad_packed_sequence(output, batch_first=True, padding_value=PAD_ID)
        agg = self._agg(output, output_lens, h_n)
        out = self._mixture(self._dropout(agg))
        return out

    @property
    def dim(self) -> int:
        """
        Returns the dimensionality of the text embedding.
        :return: Dimensionality of the text embedding
        """
        return self._out_size


class LSTMEmb(RNNEmb):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            out_size: int,
            dropout: float = 0.0,
            bidirectional: bool = True
    ):
        super().__init__(
            encoder=nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            agg=ConcatAgg([
                HiddenStateAgg(hidden_size, bidirectional),
                MaxAgg(hidden_size, bidirectional),
                AvgAgg(hidden_size, bidirectional)
            ]),
            out_size=out_size,
            dropout=dropout
        )


class GRUEmb(RNNEmb):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            out_size: int,
            dropout: float = 0.1,
            bidirectional: bool = True
    ):
        super().__init__(
            encoder=nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout,
            ),
            agg=ConcatAgg([
                HiddenStateAgg(hidden_size, bidirectional),
                MaxAgg(hidden_size, bidirectional),
                AvgAgg(hidden_size, bidirectional)
            ]),
            out_size=out_size,
            dropout=dropout
        )
