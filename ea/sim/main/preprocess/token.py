from dataclasses import dataclass, field
from typing import Generic, TypeVar, Any

T = TypeVar("T", int, str)


@dataclass
class PreTokItem:
    """
    Raw item before tokenization (for example, full method name).
    Extras contain additional token information.
    """
    value: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PostTokItem(Generic[T]):
    """
    Item after tokenization (for example, full method split by dot or BPE), could be a single value or a list of values.
    Extras contain additional token information.
    """
    value: T | list[T]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def all_ids(self) -> list[T]:
        return self.value if self.is_split else [self.value]

    @property
    def is_split(self) -> bool:
        return isinstance(self.value, list)

    def __len__(self) -> int:
        return len(self.value) if self.is_split else 0
