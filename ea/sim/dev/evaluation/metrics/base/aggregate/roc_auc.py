from sklearn.metrics import roc_auc_score

from ea.sim.dev.evaluation import MetricResult
from ea.sim.dev.evaluation.metrics.base import Candidate, Target, Score, AggregateMetric


class RocAucScore(AggregateMetric):
    def __init__(self, min_score: float):
        super().__init__("ROC_AUC")
        self._min_score = min_score

    def __call__(self, y_true: list[Target], y_pred: list[dict[Candidate, Score]]) -> MetricResult | list[MetricResult]:
        roc_auc = roc_auc_score(
            y_true=y_true,
            y_score=[-max(pred.values(), default=self._min_score) for pred in y_pred],
        )
        return [MetricResult(roc_auc, self.name)]
