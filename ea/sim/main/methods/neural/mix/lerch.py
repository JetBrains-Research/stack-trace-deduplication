from typing import Iterable

import math
import torch
from torch import nn

from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.methods.neural.similarity import Similarity
from ea.sim.main.utils import device


class TfIdfEncoder(Encoder):
    def __init__(self, vocab_size: int):
        super().__init__()
        self._vocab_size = vocab_size
        self._doc_freq = [0] * vocab_size
        self._N = 0

    def get_item_tokens(self, item: Item) -> Iterable[int]:
        for token in item.tokens:
            yield from token.all_ids

    def fit(self, items: Iterable[Item]) -> "Encoder":
        count = 0

        for item in items:
            for token_id in set(self.get_item_tokens(item)):
                self._doc_freq[token_id] += 1
            count += 1

        self._N = count
        self._doc_freq = [1 + math.log(self._N / (v + 1)) for v in self._doc_freq]
        return self

    def forward(self, items: list[Item]) -> torch.Tensor:
        vectors = torch.zeros((len(items), self._vocab_size)).to(device)
        for i, item in enumerate(items):
            for token_id in self.get_item_tokens(item):
                vectors[i, token_id] = self._doc_freq[token_id]
        return vectors

    @property
    def out_dim(self) -> int:
        return self._vocab_size


class WeightedIPSimilarity(Similarity):
    def __init__(self, vocab_size: int):
        super().__init__()
        self._weights = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self._min_value = torch.tensor(float("-inf")).to(device)
        self._norm_factor = 2_000

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        scores = (self._weights.exp() * v1 * v2).sum(dim=1)
        return scores / self._norm_factor

    @property
    def min_value(self) -> torch.Tensor:
        return self._min_value
