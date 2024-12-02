import typing as tp

from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.token import PostTokItem


class Item(tp.NamedTuple):
    tokens: list[PostTokItem[int]]

    @property
    def all_ids(self) -> list[int]:
        return [sub_token for token in self.tokens for sub_token in token.all_ids]

    def __len__(self) -> int:
        return len(self.tokens)


class ItemProcessor:
    def __init__(self, seq_coder: SeqCoder):
        self._seq_coder = seq_coder

    def __call__(self, stack_id: int) -> Item:
        return Item(tokens=self._seq_coder(stack_id))

