from dataclasses import dataclass, asdict
from typing import Any

from ea.common.evaluation import ConfidenceInterval
from ea.sim.main.utils import StackId, IssueId


@dataclass
class ScoreRecord:
    object_id: int
    score: float


@dataclass
class Prediction:
    query_id: StackId
    target_id: IssueId
    stack_scores: list[ScoreRecord]
    issue_scores: list[ScoreRecord]
    is_new_issue: bool

    @property
    def max_stack_score(self) -> float | None:
        if len(self.stack_scores) > 0:
            return max(record.score for record in self.stack_scores)

    @property
    def max_issue_score(self) -> float | None:
        if len(self.issue_scores) > 0:
            return max(record.score for record in self.issue_scores)

    @property
    def is_hit(self) -> bool:
        if len(self.issue_scores) == 0:
            return False

        top_score = max(self.issue_scores, key=lambda x: x.score)
        top_issue_id = top_score.object_id
        return top_issue_id == self.target_id

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Prediction":
        return Prediction(
            query_id=d["query_id"],
            target_id=d["target_id"],
            stack_scores=[ScoreRecord(**entry) for entry in d["stack_scores"]],
            issue_scores=[ScoreRecord(**entry) for entry in d["issue_scores"]],
            is_new_issue=d["is_new_issue"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricResult:
    value: float
    name: str
    additional: dict[str, int | float] | None = None
    interval: ConfidenceInterval | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "name": self.name,
            "additional": self.additional,
            "interval": self.interval.to_dict() if self.interval is not None else None,
        }


@dataclass
class EvaluationResult:
    results: list[MetricResult]
    name: str
    th: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "results": [x.to_dict() for x in self.results],
            "th": self.th,
        }
