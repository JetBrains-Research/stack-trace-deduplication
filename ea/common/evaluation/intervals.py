from typing import Callable

import numpy as np
from scipy.stats import beta

from ea.common.evaluation.objects import ConfidenceInterval


class Intervals:
    @staticmethod
    def binom(success: int, total: int, err: float = 0.05) -> ConfidenceInterval:
        quantile = err / 2.
        lower = beta.ppf(quantile, success, total - success + 1)
        upper = beta.ppf(1 - quantile, success + 1, total - success)
        return ConfidenceInterval(lower, upper)

    @staticmethod
    def boostrap(
            agg: Callable[[list[float]], float], scores: list[float],
            err: float = 0.05, iters: int = 100, size: int = 1
    ) -> ConfidenceInterval:
        scores = np.array(scores)
        n = len(scores)
        sn = int(size * n)
        left = int(iters * err / 2)
        values = []
        while len(values) < iters:
            inds = np.random.choice(n, sn)
            value = agg(list(scores[inds]))
            values.append(value)
        values = sorted(values)
        return ConfidenceInterval(values[left], values[iters - 1 - left])
