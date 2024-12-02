from ea.sim.dev.evaluation import Prediction, EvaluationResult
from ea.sim.dev.evaluation.metrics import Metric
from ea.sim.dev.evaluation.metrics.base import MRR


class RetrievalMRR(Metric):
    def __init__(self, ths: list[float] | None = None, boostrap: bool = False):
        super().__init__("retrieval/mrr")
        self._ths = ths
        self._metric = MRR(boostrap)

    def evaluate_with_th(self, predictions: list[Prediction], th: float = float("-inf")) -> EvaluationResult:
        mask = [not pred.is_new_issue for pred in predictions]  # evaluate only on attaches.
        y_true = [pred.target_id for flag, pred in zip(mask, predictions) if flag]
        y_pred = [
            {record.object_id: record.score for record in pred.issue_scores if record.score >= th}
            for flag, pred in zip(mask, predictions) if flag
        ]
        result = self._metric(y_true, y_pred)
        return EvaluationResult([result], self.name, th)

    def __call__(self, predictions: list[Prediction]) -> list[EvaluationResult]:
        if self._ths is None:
            return [self.evaluate_with_th(predictions)]
        return [self.evaluate_with_th(predictions, th) for th in self._ths]
