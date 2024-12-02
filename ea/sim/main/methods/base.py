from abc import abstractmethod, ABC
from typing import Iterable

from ea.sim.main.data.buckets.bucket_data import BucketData, DataSegment
from ea.sim.main.data.objects.issue import StackEvent


class SimStackModel(ABC):
    def fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'SimStackModel':
        return self.partial_fit(sim_train_data, unsup_data)

    @abstractmethod
    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'SimStackModel':
        raise NotImplementedError

    def find_params(self, sim_val_data: list[tuple[int, int, int]]) -> 'SimStackModel':
        return self

    @abstractmethod
    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        raise NotImplementedError

    def predict_pairs(self, sim_data: list[tuple[int, int, int]]) -> list[float]:
        return [self.predict(stack1, [stack2])[0] for stack1, stack2, l in sim_data]

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_score(self) -> float:
        raise NotImplementedError


class SimPairStackModel(SimStackModel):
    @abstractmethod
    def predict_pair(self, stack_id1: int, stack_id2: int) -> float:
        raise NotImplementedError

    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        return [self.predict_pair(anchor_id, stack_id) for stack_id in stack_ids]


class SimIssueModel(ABC):
    @abstractmethod
    def partial_fit(self, sim_train_data: list[tuple[int, int, int]], unsup_data: list[int] | None = None):
        raise NotImplementedError

    def find_params(self, data: BucketData, val: DataSegment) -> 'SimIssueModel':
        return self

    @abstractmethod
    def predict(self, events: Iterable[StackEvent]) -> list[tuple[int, dict[int, float]]]:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
