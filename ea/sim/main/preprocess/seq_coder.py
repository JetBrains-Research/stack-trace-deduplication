import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from ea.sim.main.data.objects.stack import Stack
from ea.sim.main.data.stack_loader import StackLoader
from ea.sim.main.preprocess.char_filter import CharFilter
from ea.sim.main.preprocess.entry_coders import Entry2Seq
from ea.sim.main.preprocess.token import PreTokItem, PostTokItem
from ea.sim.main.preprocess.tokenizers import Tokenizer, Padding


class SeqCoder:
    def __init__(
            self,
            stack_loader: StackLoader,
            entry_to_seq: Entry2Seq,
            tokenizer: Tokenizer,
            max_len: int | None = None
    ):
        self.stack_loader = stack_loader
        self.entry_to_seq = entry_to_seq
        self.char_filter = CharFilter()
        self.tokenizer = Padding(tokenizer, max_len)
        self.fitted = False
        self._name = "_".join(
            filter(
                lambda x: x.strip(),
                (entry_to_seq.name(), self.tokenizer.name())
            )
        )

    def partial_fit(self, stack_ids: list[int]) -> 'SeqCoder':
        def stack_generator() -> Iterable[Stack]:
            for stack_id in stack_ids:
                if self.stack_loader.exists(stack_id):
                    yield self.stack_loader(stack_id)

        def seq_generator() -> Iterable[list[PreTokItem]]:
            for stack in stack_generator():
                yield self.char_filter(self.entry_to_seq(stack))

        seqs = seq_generator()
        self.tokenizer.partial_fit(tqdm(seqs, desc="[SeqCoder] Fitting", total=len(stack_ids)))
        self.fitted = True
        return self

    def _pre_call(self, stack_id: int) -> list[PreTokItem]:
        res = self.stack_loader(stack_id)
        for tr in [self.entry_to_seq, self.char_filter]:
            if tr is not None:
                res = tr(res)
        return res

    def __call__(self, stack_id: int) -> list[PostTokItem[int]]:
        return self.tokenizer(self._pre_call(stack_id))

    def to_seq(self, stack_id: int) -> list[PostTokItem[str]]:
        return self.tokenizer.split(self._pre_call(stack_id))

    @lru_cache(maxsize=None)
    def to_ids(self, stack_id: int) -> list[int]:
        return [token.value for token in self(stack_id)]

    @lru_cache(maxsize=200_000)
    def ngrams(self, stack_id: int, n: int = None, ns: tuple[int, ...] = None) -> dict[tuple[int, ...], int]:
        assert (n is None) != (ns is None)  # only one is None
        if ns is None:
            ns = (n,)
        ngrams_map = {}
        ids = [token.value for token in self(stack_id)]
        l = len(ids)
        for i, token_id in enumerate(ids):
            for n in ns:
                if i + n <= l:
                    key = tuple(ids[i:i + n])
                    ngrams_map[key] = ngrams_map.get(key, 0) + 1
        return ngrams_map

    def __len__(self) -> int:
        return len(self.tokenizer)

    def state(self) -> dict[str, Any]:
        return dict(tokenizer=self.tokenizer.state())

    def load_state(self, state: dict[str, Any]):
        self.tokenizer.load_state(state["tokenizer"])

    def save(self, folder: Path):
        state = self.state()
        (folder / "seq_coder_state.json").write_text(json.dumps(state, indent=2))

    def load(self, folder: Path):
        state = json.loads((folder / "seq_coder_state.json").read_text())
        self.load_state(state)

    def name(self) -> str:
        return self._name

    def train(self, mode: bool = True):
        self.tokenizer.train(mode)
