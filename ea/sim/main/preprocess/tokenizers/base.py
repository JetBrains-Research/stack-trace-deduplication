from abc import ABC, abstractmethod
from typing import Iterable, Any

from ea.sim.main.preprocess.token import PreTokItem, PostTokItem


class Tokenizer(ABC):
    registry = {}

    def __init__(self):
        self._train = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Tokenizer.registry[cls.__name__] = cls

    @abstractmethod
    def fit(self, texts: Iterable[list[PreTokItem]]) -> 'Tokenizer':
        raise NotImplementedError

    @abstractmethod
    def partial_fit(self, texts: Iterable[list[PreTokItem]]) -> 'Tokenizer':
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: list[PreTokItem]) -> list[PostTokItem[int]]:
        raise NotImplementedError

    @abstractmethod
    def split(self, text: list[PreTokItem]) -> list[PostTokItem[str]]:
        raise NotImplementedError

    def __call__(self, text: list[PreTokItem], type: str = 'id') -> list[PostTokItem[int]] | list[PostTokItem[str]]:
        if type == 'id':
            return self.encode(text)
        else:
            return self.split(text)

    @abstractmethod
    def to_str(self, id: int) -> str:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def state(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: dict):
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "state": self.state()}

    @staticmethod
    def from_dict(state: dict[str, Any]) -> "Tokenizer":
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def token_split(self) -> bool:
        raise NotImplementedError

    def train(self, mode: bool = True):
        self._train = mode
