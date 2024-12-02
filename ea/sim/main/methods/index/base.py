from abc import ABC, abstractmethod

from ea.sim.main.utils import StackId


class Index(ABC):
    @abstractmethod
    def fit(self, stack_ids: list[StackId]) -> "Index":
        raise NotImplementedError

    @abstractmethod
    def insert(self, stack_ids: list[StackId]):
        raise NotImplementedError

    @abstractmethod
    def refit(self) -> "Index":
        raise NotImplementedError

    @abstractmethod
    def search(self, anchor_id: StackId, k: int, filter_ids: list[StackId] | None = None) -> list[StackId]:
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
