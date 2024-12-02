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


class S3M(CrossEncoder):
    name = "S3M"

    def __init__(self, evaluating: bool, vocab_size: int, token_emb_dim: int = 50, hidden_size: int = 100, dropout=0.05):
        super().__init__()

        self._embeddings = nn.Embedding(vocab_size, token_emb_dim)
        self._lstm = nn.LSTM(token_emb_dim, hidden_size, batch_first=True, bidirectional=True)

        self._head = nn.Sequential(
            nn.Linear(hidden_size * 4 + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self._dropout = nn.Dropout(dropout)

        self._cache = {} # hash -> embedding
        self._evaluating = evaluating

    def encode(self, items: list[Item]) -> torch.Tensor:
        tokens: list[list[int]] = [[token.value for token in item.tokens] for item in items]
        lens = [len(tokens) for tokens in tokens]

        # pad
        max_len = max(lens)
        padded_tokens = [tokens[i] + [0] * (max_len - lens[i]) for i in range(len(tokens))]

        # to tensor
        padded_tokens = torch.tensor(padded_tokens, device=device)

        # embeddings
        embeddings = self._embeddings(padded_tokens)

        # dropout
        embeddings = self._dropout(embeddings)

        # pack
        packed = pack_padded_sequence(embeddings, lens, batch_first=True, enforce_sorted=False)

        # take hidden representation from LSTM
        _, (hidden, _) = self._lstm(packed) 
        # hidden - [2, batch_size, hidden_size]
        hidden = hidden.permute(1, 0, 2).reshape(len(items), -1) # [batch_size, 2 * hidden_size]

        return hidden

    def cached_encode(self, items: list[Item]) -> torch.Tensor:
        items_hashes = [
            hash(tuple(token.value for token in item.tokens))
            for item in items
        ]

        items_not_in_cache = [items[i] for i in range(len(items)) if items_hashes[i] not in self._cache]
        idxs = [i for i in range(len(items)) if items_hashes[i] not in self._cache]

        if items_not_in_cache:
            hiddens = self.encode(items_not_in_cache)
            # for i in range(len(items_not_in_cache)):
            #     self._cache[items_hashes[i]] = hidden[i].cpu().detach()
            for idx, hidden in zip(idxs, hiddens):
                self._cache[items_hashes[idx]] = hidden.cpu().detach()

        return torch.stack([self._cache[hash] for hash in items_hashes])

    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        # if eval, than use cached_encode, otherwise use encode
        if self._evaluating:
            first_hidden = self.cached_encode(first)
            second_hidden = self.cached_encode(second)
        else:
            first_hidden = self.encode(first)
            second_hidden = self.encode(second)

        features = torch.cat([
            (first_hidden + second_hidden) / 2,
            first_hidden * second_hidden,
            (first_hidden - second_hidden).norm(dim=1).reshape(-1, 1)
        ], dim=1).to(device)

        return self._head(features).reshape(-1) # [batch_size]