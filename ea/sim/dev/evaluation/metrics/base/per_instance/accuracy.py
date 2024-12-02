import numpy as np

from ea.sim.dev.evaluation.metrics.base import PerInstanceMetric, Candidate, Target, Score


class Accuracy(PerInstanceMetric):
    def __init__(self, k: int | None = None, boostrap: bool = False):
        super().__init__(f"ACC_at_{k}", boostrap)
        self._k = k

    def compute(self, y_true: Target, y_pred: dict[Candidate, Score]) -> float:
        if len(y_pred) == 0:
            return 0.0

        predicted_sorted = sorted(y_pred.items(), key=lambda x: x[1], reverse=True)
        found_at = None
        for i, (candidate, _) in enumerate(predicted_sorted):
            if candidate == y_true:
                found_at = i
                break

        is_candidate_found = found_at is not None
        does_position_satisfy = is_candidate_found and ((self._k is None) or (found_at < self._k))
        return does_position_satisfy

    def aggregate(self, scores: list[float]) -> float:
        return float(np.mean(scores))
