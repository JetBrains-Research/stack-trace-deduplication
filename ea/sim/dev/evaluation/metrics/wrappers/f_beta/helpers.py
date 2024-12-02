from abc import ABC, abstractmethod

from ea.sim.dev.evaluation import Prediction, MetricResult
from ea.sim.dev.evaluation.metrics.base.aggregate.f_beta import ConfusionMatrix


class UpdateRule(ABC):
    def __init__(self, reverse: bool):
        self._reverse = reverse

    @abstractmethod
    def get_matrix(self, predictions: list[Prediction]) -> ConfusionMatrix:
        raise NotImplementedError

    @abstractmethod
    def update(self, prediction: Prediction, matrix: ConfusionMatrix) -> None:
        raise NotImplementedError


class ClassicUpdateRule(UpdateRule):
    def get_matrix(self, predictions: list[Prediction]) -> ConfusionMatrix:
        true_positives = sum(pred.is_new_issue for pred in predictions)
        false_positives = len(predictions) - true_positives
        if not self._reverse:
            # All are positives
            return ConfusionMatrix(TP=true_positives, FP=false_positives, TN=0, FN=0)
        else:
            # All are negatives
            return ConfusionMatrix(TP=0, FP=0, TN=false_positives, FN=true_positives)

    def update(self, prediction: Prediction, matrix: ConfusionMatrix) -> None:
        if not self._reverse:
            if not prediction.is_new_issue:
                matrix.FP -= 1
                matrix.TN += 1
            else:
                matrix.TP -= 1
                matrix.FN += 1
        else:
            if not prediction.is_new_issue:
                matrix.TN -= 1
                matrix.FP += 1
            else:
                matrix.FN -= 1
                matrix.TP += 1


class ImprovedUpdateRule(UpdateRule):
    def get_matrix(self, predictions: list[Prediction]) -> ConfusionMatrix:
        if not self._reverse:
            # All are positives
            true_positives = sum(pred.is_new_issue for pred in predictions)
            false_positives = len(predictions) - true_positives
            return ConfusionMatrix(TP=true_positives, FP=false_positives, TN=0, FN=0)
        else:
            # All are negatives
            true_negatives = sum((not pred.is_new_issue) and pred.is_hit for pred in predictions)
            false_negatives = len(predictions) - true_negatives
            return ConfusionMatrix(TP=0, FP=0, TN=true_negatives, FN=false_negatives)

    def update(self, prediction: Prediction, matrix: ConfusionMatrix) -> None:
        if not self._reverse:
            # class changed from 1 to 0.
            if prediction.is_new_issue:
                matrix.TP -= 1
            else:
                matrix.FP -= 1

            if (not prediction.is_new_issue) and prediction.is_hit:
                matrix.TN += 1
            else:
                matrix.FN += 1
        else:
            # class changed from 0 to 1.
            if (not prediction.is_new_issue) and prediction.is_hit:
                matrix.TN -= 1
            else:
                matrix.FN -= 1

            if prediction.is_new_issue:
                matrix.TP += 1
            else:
                matrix.FP += 1


class FBetaAll:
    def __init__(self, update_rule: UpdateRule, beta: float = 1.0, reverse: bool = False):
        self._update_rule = update_rule
        self._beta = beta
        self._reverse = reverse
        self._name = f"F_{beta}"

    def __call__(self, predictions: list[Prediction]) -> list[MetricResult]:
        predictions = [pred for pred in predictions if pred.max_issue_score is not None]
        predictions = sorted(predictions, key=lambda x: x.max_issue_score)
        matrix = self._update_rule.get_matrix(predictions)
        results = []
        for prediction in predictions:
            self._update_rule.update(prediction, matrix)
            f_beta = matrix.f_beta(self._beta)
            if f_beta is not None:
                result = MetricResult(
                    f_beta, self._name,
                    additional=matrix.additional | {"th": prediction.max_issue_score}
                )
                results.append(result)
        return results


class FBeta:
    def __init__(self, update_rule: UpdateRule, beta: float = 1.0, reverse: bool = False):
        self._f_beta_all = FBetaAll(update_rule, beta, reverse)

    def __call__(self, predictions: list[Prediction]) -> MetricResult:
        results = self._f_beta_all(predictions)
        best_result = max(results, key=lambda x: x.value)
        return best_result
