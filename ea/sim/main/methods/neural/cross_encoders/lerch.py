import math
from typing import Iterable

import torch

from ea.sim.main.methods.neural.cross_encoders.base import CrossEncoder
from ea.sim.main.methods.neural.encoders.objects import Item


class LerchCrossEncoder(CrossEncoder):
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self._vocab_size = vocab_size
        self._doc_freq = [0] * vocab_size
        self._N = 0

    def get_item_tokens(self, item: Item) -> Iterable[int]:
        for token in item.tokens:
            yield from token.all_ids

    def fit(self, items: Iterable[Item]) -> "LerchCrossEncoder":
        # doc_freq[t] = idf of token t
        count = 0

        for item in items:
            for token_id in set(self.get_item_tokens(item)):
                self._doc_freq[token_id] += 1
            count += 1

        self._N = count
        self._doc_freq = [1 + math.log(self._N / (v + 1)) for v in self._doc_freq]
        return self


    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        assert len(first) == len(second)

        scores = []

        for i in range(len(first)):
            tokens_first = list(self.get_item_tokens(first[i]))
            tokens_second = list(self.get_item_tokens(second[i]))

            # score = sum_{common token t} idf(t) ^ 2
            
            score = 0
            for token in set(tokens_first) & set(tokens_second):
                score += self._doc_freq[token] ** 2
            scores.append(score)
        
        return torch.tensor(scores)
