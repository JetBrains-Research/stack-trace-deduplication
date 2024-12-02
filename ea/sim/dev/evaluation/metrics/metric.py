from abc import ABC, abstractmethod

from ea.sim.dev.evaluation import Prediction, EvaluationResult


class Metric(ABC):
    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def __call__(self, predictions: list[Prediction]) -> list[EvaluationResult]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self._name
