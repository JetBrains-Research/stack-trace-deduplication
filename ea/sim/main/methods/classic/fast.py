from collections import Counter
from typing import NamedTuple

from math import exp

from ea.sim.main.exceptions import NotFittedError, AlreadyFittedError
from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.utils import StackId


class _Token(NamedTuple):
    id: int
    pos: int


class _Index:
    def __init__(self, coder: SeqCoder):
        self._coder = coder
        self._total_traces = 0
        self._token_freq = Counter()

    def fit(self, unsup_data: list[StackId]) -> "_Index":
        for stack_id in unsup_data:
            token_ids = self._coder.to_ids(stack_id)
            for token_id in set(token_ids):
                self._token_freq[token_id] += 1
            self._total_traces += 1
        return self

    def traces_with_frame(self, id: int) -> int:
        return self._token_freq[id]

    @property
    def total_traces(self) -> int:
        return self._total_traces


class _AlignmentAlgorithm:
    max_score = 1.0
    min_score = -1.0

    def __init__(self, index: _Index, alpha: float, beta: float, gamma: float):
        self._index = index
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    def compute(self, query: list[_Token], doc: list[_Token]) -> float:
        i = j = 0
        sim = 0.0

        while i < len(query) and j < len(doc):
            query_token, doc_token = query[i], doc[j]
            if query_token.id == doc_token.id:
                sim += self.match(query_token, doc_token)
                i += 1
                j += 1
            elif query_token.id < doc_token.id:
                sim -= self.gap(query_token)
                i += 1
            else:
                sim -= self.gap(doc_token)
                j += 1

        sim -= sum(self.gap(token) for token in query[i:])
        sim -= sum(self.gap(token) for token in doc[j:])
        return self.normalize(sim, query, doc)

    def normalize(self, score: float, query: list[_Token], doc: list[_Token]) -> float:
        query_norm = sum(self.weight(x) for x in query)
        doc_norm = sum(self.weight(x) for x in doc)
        if (query_norm == 0) and (doc_norm == 0):
            # Both traces are empty.
            return _AlignmentAlgorithm.max_score
        return score / (query_norm + doc_norm)

    def weight(self, x: _Token) -> float:
        op_1 = 1 / (x.pos ** self._alpha)
        op_2 = exp(-self._beta * self._index.traces_with_frame(x.id) / self._index.total_traces)
        return op_1 * op_2

    def gap(self, x: _Token) -> float:
        return self.weight(x)

    def match(self, x1: _Token, x2: _Token) -> float:
        diff = exp(-self._gamma * abs(x1.pos - x2.pos))
        return (self.weight(x1) + self.weight(x2)) * diff


class FaST(SimStackModel):
    # Implementation: https://irving-muller.github.io/papers/FaST.pdf

    def __init__(self, coder: SeqCoder, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0):
        self._index = _Index(coder)
        self._algorithm = _AlignmentAlgorithm(self._index, alpha, beta, gamma)
        self._coder = coder
        self._fitted = False

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'SimStackModel':
        if self._fitted:
            raise AlreadyFittedError()
        self._index.fit(unsup_data)
        self._fitted = True
        return self

    def _predict_pair(self, query_id: int, doc_id: int) -> float:
        def to_tokens(stack_id: int) -> list[_Token]:
            tokens = self._coder.to_ids(stack_id)
            tokens = tokens[::-1]
            tokens = [_Token(id=token_id, pos=pos) for pos, token_id in enumerate(tokens, start=1)]
            tokens = sorted(tokens, key=lambda token: token.id)
            return tokens

        query_tokens = to_tokens(query_id)
        doc_tokens = to_tokens(doc_id)
        return self._algorithm.compute(query_tokens, doc_tokens)

    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        if not self._fitted:
            raise NotFittedError()
        return [self._predict_pair(anchor_id, stack_id) for stack_id in stack_ids]

    def name(self) -> str:
        return self._coder.name() + "_FaST"

    @property
    def min_score(self) -> float:
        return _AlignmentAlgorithm.min_score
