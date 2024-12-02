import typing as tp
from typing import Any, Iterable


class SpecialToken(tp.NamedTuple):
    id: int
    value: str


class SpecialTokens:
    PAD: SpecialToken = SpecialToken(0, "[PAD]")
    UNK: SpecialToken = SpecialToken(1, "[UNK]")
    SOS: SpecialToken = SpecialToken(2, "[SOS]")
    EOS: SpecialToken = SpecialToken(3, "[EOS]")

    all = [PAD, UNK, SOS, EOS]

    @staticmethod
    def to_str(id: int) -> str:
        for token in SpecialTokens.all:
            if id == token.id:
                return token.value


class IdCoder:
    def __init__(self):
        self.id2name = {token.id: token.value for token in SpecialTokens.all}
        self.name2id = {token.value: token.id for token in SpecialTokens.all}
        self.fixed = False

    def encode(self, word: Any) -> int:
        if (not self.fixed) and (word not in self.name2id):
            self.name2id[word] = len(self.name2id)
            self.id2name[self.name2id[word]] = word
        return self.name2id.get(word, SpecialTokens.UNK.id)

    def __getitem__(self, item: Any) -> int:
        return self.encode(item)

    def encodes(self, words: Iterable[Any]) -> list[int]:
        return [self.encode(word) for word in words]

    def decode(self, id: int) -> Any:
        return self.id2name[id]

    def decodes(self, ids: Iterable[int]) -> list[Any]:
        return [self.decode(id) for id in ids]

    def state(self) -> dict[str, int]:
        return dict(fixed=self.fixed, vocab=self.name2id)

    def load_state(self, state: dict[str, int]):
        self.name2id = state["vocab"]
        self.id2name = {i: t for t, i in state["vocab"].items()}
        self.fixed = state["fixed"]

    def fix(self, fixed: bool = True):
        self.fixed = fixed

    def __len__(self) -> int:
        return len(self.name2id)
