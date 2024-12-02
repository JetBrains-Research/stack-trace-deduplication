import re
from abc import ABC, abstractmethod


class CrashFrameParser(ABC):
    @abstractmethod
    def search(self, x: str) -> str | None:
        raise NotImplementedError

    def satisfy(self, x: str) -> bool:
        return self.search(x) is not None

    def parse(self, x: str) -> str | None:
        return self.search(x)


class LibFrameParser(CrashFrameParser):
    """
    Parses library information from a frame.
    As usual, it is at the beginning of the line in the square brackets.
    For example, for line '[libsystem_pthread.dylib+0x726c]  _pthread_start+0x94' should return 'libsystem_pthread.dylib'.
    """

    _regexp = re.compile("^\[\S+\+0x[a-f0-9]+]")  # [...]

    def search(self, x: str) -> str | None:
        x = x.strip()
        result = LibFrameParser._regexp.search(x)

        if result is not None:
            group = result.group()
            last_pos = group.find("+")  # all until "+"
            return group[1:last_pos]


class MethodFrameParser(CrashFrameParser):
    """
    Parses method information from a frame.
    """

    _regexp = re.compile("(?:[a-zA-Z]+[a-zA-Z0-9_]+\.){2,}[a-zA-Z0-9_]{2,}")

    def search(self, x: str) -> str | None:
        x = x.strip()
        result = MethodFrameParser._regexp.search(x)
        if result is not None:
            return result.group()
