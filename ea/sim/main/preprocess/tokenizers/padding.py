from typing import Any, Iterable

from ea.sim.main.preprocess.id_coder import SpecialTokens
from ea.sim.main.preprocess.token import PreTokItem, PostTokItem, T
from ea.sim.main.preprocess.tokenizers import Tokenizer


class Padding(Tokenizer):
    def __init__(self, tokenizer: Tokenizer, max_len: int | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def fit(self, texts: Iterable[list[PreTokItem]]) -> 'Padding':
        self.tokenizer.fit(texts)
        return self

    def partial_fit(self, texts: Iterable[list[PreTokItem]]) -> 'Padding':
        self.tokenizer.partial_fit(texts)
        return self

    def pad_seq(self, seq: list[Any], pad: Any) -> list[Any]:
        if self.max_len is not None:
            return seq[len(seq) - min(len(seq), self.max_len):]
            # if len(seq) < self.max_len - 1:
            #     return [pad] * (self.max_len - 1 - len(seq)) + seq
            # else:
            #     return seq[len(seq) - min(len(seq), self.max_len):]
        return seq

    def special_tok(self, value: T) -> PostTokItem[T]:
        return PostTokItem([value]) if self.tokenizer.token_split else PostTokItem(value)

    def encode(self, text: list[PreTokItem]) -> list[PostTokItem[int]]:
        enc_seq = self.tokenizer.encode(text)
        out_seq = self.pad_seq(enc_seq, self.special_tok(SpecialTokens.PAD.id))
        out_seq = [self.special_tok(SpecialTokens.SOS.id)] + out_seq
        out_seq = out_seq + [self.special_tok(SpecialTokens.EOS.id)]
        return out_seq

    def split(self, text: list[PreTokItem]) -> list[PostTokItem[str]]:
        out_seq = self.pad_seq(self.tokenizer.split(text), self.special_tok(SpecialTokens.PAD.value))
        out_seq = [self.special_tok(SpecialTokens.SOS.value)] + out_seq
        out_seq = out_seq + [self.special_tok(SpecialTokens.EOS.value)]
        return out_seq

    def to_str(self, id: int) -> str:
        if id >= len(SpecialTokens.all):
            return self.tokenizer.to_str(id)
        else:
            return SpecialTokens.to_str(id)

    def state(self) -> dict[str, Any]:
        return dict(tokenizer=self.tokenizer.state(), max_len=self.max_len)

    def load_state(self, state: dict):
        self.tokenizer.load_state(state["tokenizer"])
        self.max_len = state["max_len"]

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Padding":
        return Padding(Tokenizer.from_dict(state["tokenizer"]), state["max_len"])

    def __len__(self) -> int:
        return len(self.tokenizer)

    def name(self) -> str:
        return self.tokenizer.name() + (f"_pad{self.max_len}" if self.max_len is not None else "")

    @property
    def token_split(self) -> bool:
        return self.tokenizer.token_split

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.train(mode)
