from abc import abstractmethod
from typing import Iterable


class IssueScorer:
    @abstractmethod
    def score(self, scores: Iterable[float], with_arg: bool = False) -> float | tuple[float, int]:
        raise NotImplementedError

    @property
    @abstractmethod
    def support_dup_removal(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
