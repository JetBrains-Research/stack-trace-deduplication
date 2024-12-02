from dataclasses import dataclass
from typing import NamedTuple, Any


class Frame(NamedTuple):
    name: str
    file_name: str | None = None
    line_number: int | None = None
    commit_hash: str | None = None
    subsystem: str | None = None
    has_marker: bool = False
    timestamp: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "file_name": self.file_name,
            "line_number": self.line_number,
            "commit_hash": self.commit_hash,
            "subsystem": self.subsystem
        }


@dataclass(frozen=True)
class Stack:
    id: int
    timestamp: int
    errors: list[str] | None
    frames: list[Frame]
    messages: list[str] | None
    comment: list[str] | None
    issue_id: int = -1

    def eq_content(self, stack: 'Stack') -> bool:
        return self.errors == stack.errors and \
            self.frames == stack.frames and \
            self.messages == stack.messages and \
            self.comment == stack.comment

    @property
    def is_soe(self) -> bool:
        return self.errors and any('StackOverflow' in cl for cl in self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "messages": self.messages,
            "elements": [frame.to_dict() for frame in self.frames],
            "comment": self.comment
        }

    def __hash__(self) -> int:
        # Required for identify duplicates.
        methods = ",".join([frame.name for frame in self.frames])
        return hash(methods)
