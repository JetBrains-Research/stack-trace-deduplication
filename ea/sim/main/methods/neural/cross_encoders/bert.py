from dataclasses import dataclass
import torch
import torch.nn as nn

from ea.sim.main.methods.neural.cross_encoders.base import CrossEncoder
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.tokens import BOWTokenEncoder
from ea.sim.main.utils import device

@dataclass
class BertEncoderConfig:
    d_input: int = 50
    d_model: int = 100
    nheads: int = 4
    nlayers: int = 3
    n_ctx_each_side: int = 100
    dropout: float = 0.0

    vocab_size: int = 10000

    def __str__(self):
        return f"{self.d_input}_{self.d_model}_{self.nheads}_{self.nlayers}_{self.n_ctx_each_side}_{self.dropout}"


class PositionalEncoding(nn.Module):
    def __init__(self, config: BertEncoderConfig):
        super().__init__()
        self._config = config
        self._embs = nn.Embedding(config.n_ctx_each_side, config.d_input)
        self._embs_sequence = nn.Embedding(2, config.d_input)
    
    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        """
        Return tensor to add to input embeddings in order to add positional information and 
        distinguish between first and second sequence.
        :param first: Sequence of N items
        :param second: Sequence of N items
        :return: Tensor of size (N, seq_len, d_input)
        """

        first_lens = [min(len(item.tokens), self._config.n_ctx_each_side) for item in first]
        second_lens = [min(len(item.tokens), self._config.n_ctx_each_side) for item in second]

        positions = [
            list(range(first_len)) + 
            list(range(second_len)) + 
            [0] * (self._config.n_ctx_each_side * 2 - first_len - second_len) 
            for first_len, second_len in zip(first_lens, second_lens)
        ]
        positions = torch.tensor(positions).to(device) # (N, seq_len)
        positions_embs = self._embs(positions) # (N, seq_len, d_input)

        sequence_embs = [
            [0] * first_len + 
            [1] * second_len + 
            [0] * (self._config.n_ctx_each_side * 2 - first_len - second_len) 
            for first_len, second_len in zip(first_lens, second_lens)
        ]
        sequence_embs = torch.tensor(sequence_embs).to(device) # (N, seq_len)
        sequence_embs = self._embs_sequence(sequence_embs) # (N, seq_len, d_input)

        return positions_embs + sequence_embs


class BertEncoder(CrossEncoder):
    def __init__(self, config: BertEncoderConfig):
        super().__init__()
        self._config = config
        self.model = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nheads,
                dim_feedforward=config.d_model * 4,
                dropout=config.dropout,
                batch_first=True
            ),
            num_layers=config.nlayers
        )
        self._class_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))
        self._sep_embedding = nn.Parameter(torch.randn(1, 1, config.d_model))

        self._positional_encoding = PositionalEncoding(config)
        self._token_encoder = BOWTokenEncoder(config.vocab_size, config.d_input, config.dropout, max_seq_len=config.n_ctx_each_side * 2)
        # self._token_encoder = RNNTokenEncoder(config.vocab_size, config.d_input, config.dropout)

        self._lin_inp_to_model = nn.Linear(config.d_input, config.d_model)
        self._model_to_score = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1)
        )
        self._last_activation = nn.Tanh()

    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        """
        Return similarity scores for each pair of items. 
        :param first: Sequence of N items
        :param second: Sequence of N items
        :return: Tensor of size (N, 1)
        """

        concatenated = [first_item.tokens[-self._config.n_ctx_each_side:] + second_item.tokens[-self._config.n_ctx_each_side:]
                        for first_item, second_item in zip(first, second)]

        embs, mask = self._token_encoder(concatenated)

        embs = self._positional_encoding(first, second) + embs # (batch_size, seq_len, d_input)

        embs = self._lin_inp_to_model(embs) # (batch_size, seq_len, d_model)
        padding_mask = mask.all(dim=2)

        # adding class embedding
        embs = torch.cat([self._class_embedding.repeat(embs.shape[0], 1, 1), embs], dim=1)
        padding_mask = torch.cat([torch.ones(padding_mask.shape[0], 1, dtype=torch.bool).to(device), padding_mask], dim=1)

        out = self.model(embs, src_key_padding_mask=padding_mask) # (batch_size, seq_len, d_model)

        scores = self._model_to_score(out[:, 0]) # (batch_size, 1)
        scores = self._last_activation(scores)

        return scores.squeeze(-1)
