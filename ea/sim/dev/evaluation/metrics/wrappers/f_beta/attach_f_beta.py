from ea.sim.dev.evaluation import Prediction, EvaluationResult
from ea.sim.dev.evaluation.metrics import Metric
from ea.sim.dev.evaluation.metrics.wrappers.f_beta.helpers import FBeta, ClassicUpdateRule


class AttachFBeta(Metric):
    def __init__(self, betas: list[float]):
        super().__init__("attach/f_beta")
        self._betas = betas
        self._metrics = [FBeta(ClassicUpdateRule(reverse=True), beta, reverse=True) for beta in betas]

    def __call__(self, predictions: list[Prediction]) -> list[EvaluationResult]:
        predictions = [pred for pred in predictions if pred.max_issue_score is not None]
        metrics = [metric(predictions) for metric in self._metrics]
        return [EvaluationResult(metrics, self.name)]
