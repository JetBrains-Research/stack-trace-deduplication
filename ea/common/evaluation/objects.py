from typing import TypeAlias, NamedTuple

MetricValue: TypeAlias = float


class ConfidenceInterval(NamedTuple):
    left: float
    right: float

    def to_dict(self) -> dict[str, float]:
        return {
            "left": self.left,
            "right": self.right,
        }
