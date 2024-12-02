from typing import Iterable

from ea.sim.main.methods.issue_scorer import IssueScorer


class MaxIssueScorer(IssueScorer):
    def score(self, scores: Iterable[float], with_arg: bool = False) -> float | tuple[float, int]:
        if with_arg:
            ind = 0
            value = None
            for i, score in enumerate(scores):
                if value is None or score > value:
                    ind = i
                    value = score
            return value, ind
        return max(scores)

    @property
    def support_dup_removal(self) -> bool:
        # Supports duplicate removing because the score depends on objects with max score.
        return True

    def name(self) -> str:
        return "max_scorer"
