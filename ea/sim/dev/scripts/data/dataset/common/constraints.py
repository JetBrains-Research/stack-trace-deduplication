"""
Constraints for processed issues / reports. For example, "predefined issues" are removed from triplet training.
"""
import json
from abc import abstractmethod, ABC
from pathlib import Path


class Constraints(ABC):
    @abstractmethod
    def is_sorted_issue(self, issue_id: int) -> bool:
        raise NotImplementedError

    @staticmethod
    def load(path: Path | None = None) -> "Constraints":
        if path is None:
            return DefaultConstraints()
        return FileBasedConstraints.from_file(path)


class DefaultConstraints(Constraints):
    def is_sorted_issue(self, issue_id: int) -> bool:
        return True


class FileBasedConstraints(Constraints):
    def __init__(self, predefined_issues: set[int]):
        self._predefined_issues = predefined_issues

    def is_sorted_issue(self, issue_id: int) -> bool:
        return issue_id not in self._predefined_issues

    @staticmethod
    def from_file(path: Path) -> Constraints:
        constraints = json.load(path.expanduser().open("r"))
        return FileBasedConstraints(
            predefined_issues=constraints["predefined_issues"]
        )
