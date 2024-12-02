from abc import ABC, abstractmethod
from typing import TypeAlias

from ea.common.evaluation import Intervals
from ea.sim.dev.evaluation import MetricResult

Target: TypeAlias = int
Candidate: TypeAlias = int
Score: TypeAlias = float


class BaseMetric(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> MetricResult | list[MetricResult]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name


class PerInstanceMetric(BaseMetric):
    def __init__(self, name: str, boostrap: bool):
        super().__init__(name)
        self._boostrap = boostrap

    @abstractmethod
    def compute(self, y_true: Target, y_pred: dict[Candidate, Score]) -> float:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, scores: list[float]) -> float:
        raise NotImplementedError

    def computes(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> list[float]:
        return [self.compute(true, pred) for true, pred in zip(y_true, y_pred)]

    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> MetricResult | list[MetricResult]:
        scores = self.computes(y_true, y_pred)
        value = self.aggregate(scores)
        interval = Intervals.boostrap(self.aggregate, scores) if self._boostrap else None
        return MetricResult(
            value=value,
            name=self.name,
            interval=interval
        )


class AggregateMetric(BaseMetric):
    @abstractmethod
    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> MetricResult | list[MetricResult]:
        raise NotImplementedError
