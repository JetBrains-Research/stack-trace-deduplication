from dataclasses import dataclass

import numpy as np

from ea.sim.dev.evaluation import MetricResult
from ea.sim.dev.evaluation.metrics.base import Candidate, Target, Score, AggregateMetric


@dataclass
class ConfusionMatrix:
    TP: int
    FP: int
    TN: int
    FN: int

    @property
    def precision(self) -> float | None:
        denom = self.TP + self.FP
        return None if denom == 0 else self.TP / denom

    @property
    def recall(self) -> float | None:
        denom = self.TP + self.FN
        return None if denom == 0 else self.TP / denom

    def f_beta(self, beta: float) -> float | None:
        if (self.precision is not None) and (self.recall is not None) and (self.precision > 0 or self.recall > 0):
            beta_squared = beta ** 2
            return (1 + beta_squared) * (self.precision * self.recall) / ((beta_squared * self.precision) + self.recall)

    @property
    def additional(self) -> dict:
        return self.__dict__ | {"Precision": self.precision, "Recall": self.recall}

    @staticmethod
    def with_min_score(n_pos: int, n_neg: int, reverse: bool) -> "ConfusionMatrix":
        # All objects relate to positives.
        if not reverse:
            return ConfusionMatrix(TP=n_pos, FP=n_neg, TN=0, FN=0)
        else:
            return ConfusionMatrix(TP=0, FP=0, TN=n_neg, FN=n_pos)


class FBetaAll(AggregateMetric):
    """
    Iteratively computes all F-beta scores.
    """

    def __init__(self, beta: float = 1.0, reverse: bool = False):
        super().__init__(f"F_{beta}_all")
        self._beta = beta
        self._reverse = reverse  # objects with the lowest score are positives.

    def update(self, matrix: ConfusionMatrix, target: int) -> None:
        if not self._reverse:
            if target == 0:
                matrix.FP -= 1
                matrix.TN += 1
            elif target == 1:
                matrix.TP -= 1
                matrix.FN += 1
            else:
                raise ValueError("Target must be 0 or 1.")
        else:
            if target == 0:
                matrix.TN -= 1
                matrix.FP += 1
            elif target == 1:
                matrix.FN -= 1
                matrix.TP += 1
            else:
                raise ValueError("Target must be 0 or 1.")

    def calculate(self, y_true: list[Target], y_score: list[Score]) -> list[MetricResult]:
        sort_mask = np.argsort(y_score)
        y_true = np.array(y_true)[sort_mask]
        y_score = np.array(y_score)[sort_mask]
        n_pos, n_neg = int(sum(y_true == 1)), int(sum(y_true == 0))
        matrix = ConfusionMatrix.with_min_score(n_pos, n_neg, self._reverse)
        results = []
        for true, score in zip(y_true, y_score):
            f_beta = matrix.f_beta(self._beta)
            self.update(matrix, true)
            if f_beta is not None:
                result = MetricResult(f_beta, self.name, additional=matrix.additional | {"th": score})
                results.append(result)
        return results

    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> list[MetricResult]:
        y_score = [max(pred.items(), key=lambda x: x[1])[1] for pred in y_pred]
        return self.calculate(y_true, y_score)


class FBeta(AggregateMetric):
    """
    Computes best F-beta score with auxiliary metrics.
    """

    def __init__(self, beta: float = 1.0, reverse: bool = False):
        super().__init__(f"F_{beta}")
        self._f_beta_all = FBetaAll(beta=beta, reverse=reverse)

    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> MetricResult:
        metric_results = self._f_beta_all(y_true, y_pred)
        best_result = max(metric_results, key=lambda x: x.value)
        return best_result
