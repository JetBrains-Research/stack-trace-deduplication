from ea.sim.dev.evaluation import Prediction, EvaluationResult
from ea.sim.dev.evaluation.metrics import Metric
from ea.sim.dev.evaluation.metrics.base import Accuracy


class RetrievalAccuracy(Metric):
    def __init__(self, ks: list[int], ths: list[float] | None = None, boostrap: bool = False):
        super().__init__("retrieval/accuracy")
        self._metrics = [Accuracy(k, boostrap) for k in ks]
        self._ths = ths

    def evaluate_with_th(self, predictions: list[Prediction], th: float = float("-inf")) -> EvaluationResult:
        mask = [not pred.is_new_issue for pred in predictions]  # evaluate only on attaches.
        y_true = [pred.target_id for flag, pred in zip(mask, predictions) if flag]
        y_pred = [
            {record.object_id: record.score for record in pred.issue_scores if record.score >= th}
            for flag, pred in zip(mask, predictions) if flag
        ]
        results = [metric(y_true, y_pred) for metric in self._metrics]
        return EvaluationResult(results, self.name, th)

    def __call__(self, predictions: list[Prediction]) -> list[EvaluationResult]:
        if self._ths is None:
            return [self.evaluate_with_th(predictions)]
        return [self.evaluate_with_th(predictions, th) for th in self._ths]
