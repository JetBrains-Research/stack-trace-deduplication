from typing import Any, Iterable

from loguru import logger

from ea.sim.main.preprocess.id_coder import IdCoder
from ea.sim.main.preprocess.token import PreTokItem, PostTokItem
from ea.sim.main.preprocess.tokenizers import Tokenizer


class SimpleTokenizer(Tokenizer):
    def __init__(self, split_sep: str | None = None):
        super().__init__()
        self.coder = IdCoder()
        self.split_sep = split_sep

    def _split_pre_tok(self, pre_tok: PreTokItem) -> str | list[str]:
        if self.split_sep is not None:
            return pre_tok.value.split(self.split_sep)
        else:
            return pre_tok.value

    def _encode_pre_tok(self, pre_tok: PreTokItem) -> int | list[int]:
        pre_tok_split = self._split_pre_tok(pre_tok)
        if isinstance(pre_tok_split, list):
            return self.coder.encodes(pre_tok_split)
        else:
            return self.coder.encode(pre_tok_split)

    def _fit_coder(self, texts: Iterable[list[PreTokItem]]) -> "SimpleTokenizer":
        for text in texts:
            for token in text:
                self._encode_pre_tok(token)
        return self

    def fit(self, texts: Iterable[list[PreTokItem]]) -> "SimpleTokenizer":
        self._fit_coder(texts)
        self.coder.fix()
        return self

    def partial_fit(self, texts: Iterable[list[PreTokItem]]) -> "SimpleTokenizer":
        self.coder.fix(False)
        self._fit_coder(texts)
        self.coder.fix()
        logger.debug(f"Tokenizer fitted, {len(self.coder)} coder size.")
        return self

    def encode(self, text: list[PreTokItem]) -> list[PostTokItem[int]]:
        return [
            PostTokItem(value=self._encode_pre_tok(token), extras=token.extras)
            for token in text
        ]

    def split(self, text: list[PreTokItem]) -> list[PostTokItem[str]]:
        return [
            PostTokItem(value=self._split_pre_tok(token), extras=token.extras)
            for token in text
        ]

    def to_str(self, id: int) -> str:
        return self.coder.decode(id)

    def state(self) -> dict[str, Any]:
        return dict(coder=self.coder.state())

    def load_state(self, state: dict):
        self.coder.load_state(state["coder"])

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "SimpleTokenizer":
        tokenizer = SimpleTokenizer()
        tokenizer.coder = IdCoder().load_state(state["coder"])
        return tokenizer

    def __len__(self):
        return len(self.coder)

    def name(self) -> str:
        return "simple"

    @property
    def token_split(self) -> bool:
        return self.split_sep is not None
