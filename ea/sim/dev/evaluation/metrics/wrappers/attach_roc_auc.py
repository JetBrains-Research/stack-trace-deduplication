from ea.sim.dev.evaluation import Prediction, EvaluationResult
from ea.sim.dev.evaluation.metrics import Metric
from ea.sim.dev.evaluation.metrics.base import RocAucScore


class AttachRocAuc(Metric):
    def __init__(self, min_score: float = -1):
        super().__init__("attach/roc_auc")
        self._metric = RocAucScore(min_score)

    def __call__(self, predictions: list[Prediction]) -> list[EvaluationResult]:
        predictions = [pred for pred in predictions if pred.max_issue_score is not None]
        y_true = [int(pred.is_new_issue) for pred in predictions]
        predictions_dict = [{record.object_id: record.score for record in pred.issue_scores} for pred in predictions]
        metric = self._metric(y_true, predictions_dict)
        return [EvaluationResult(metric, self.name)]
