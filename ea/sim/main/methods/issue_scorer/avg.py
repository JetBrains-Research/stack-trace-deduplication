from typing import Iterable

from ea.sim.main.methods.issue_scorer import IssueScorer


class AvgIssueScorer(IssueScorer):
    def score(self, scores: Iterable[float], with_arg: bool = False) -> float | tuple[float, int]:
        if with_arg:
            raise ValueError("Avg scorer not support index")
        s, c = 0, 0
        for value in scores:
            s += value
            c += 1
        return s / c

    @property
    def support_dup_removal(self) -> bool:
        # Does not support duplicate removing because the score depends on all candidates.
        return False

    def name(self) -> str:
        return "avg_scorer"
