import math
import numpy as np

from ea.sim.dev.evaluation.metrics.base import PerInstanceMetric, Candidate, Target, Score


class MRR(PerInstanceMetric):
    def __init__(self, boostrap: bool = False):
        super().__init__("MRR", boostrap)

    def compute(self, y_true: Target, y_pred: dict[Candidate, Score]) -> float:
        # Reciprocal rank.
        if len(y_pred) == 0:
            return 0.0

        predicted_sorted = sorted(y_pred.items(), key=lambda x: x[1], reverse=True)
        found_at = math.inf
        for i, (candidate, _) in enumerate(predicted_sorted):
            if candidate == y_true:
                found_at = i
                break

        return 1. / (found_at + 1)

    def aggregate(self, scores: list[float]) -> float:
        return float(np.mean(scores))
