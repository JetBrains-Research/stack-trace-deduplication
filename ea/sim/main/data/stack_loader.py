from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from loguru import logger

from ea.sim.main.data.objects.stack import Stack
from ea.sim.main.data.parsers.parser_v1 import StackParserV1


class StackLoader(ABC):
    parser = StackParserV1()

    @abstractmethod
    def exists(self, stack_id: int) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, stack_id: int) -> Stack | None:
        raise NotImplementedError

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError


class DirectoryStackLoader(StackLoader):
    file_name_mask = r"[0-9]*.json"

    def __init__(self, folder: Path, frames_field: str = 'elements', st_issue: dict[int, int] | None = None):
        self.folder = folder.expanduser()

        self.sid_to_dir = {}
        for file_path in self.folder.rglob(self.file_name_mask):
            stack_id, _ = file_path.name.split(".")
            self.sid_to_dir[int(stack_id)] = file_path.parent

        logger.debug(f"Found {len(self.sid_to_dir)} stacks")

        self.frames_field = frames_field
        self.st_issue = st_issue

    def exists(self, stack_id: int) -> bool:
        return stack_id in self.sid_to_dir

    @lru_cache(maxsize=300_000)
    def __call__(self, stack_id: int) -> Stack | None:
        if not self.exists(stack_id):
            return

        issue_id = -1
        if self.st_issue is not None:
            issue_id = self.st_issue[stack_id]
            if issue_id == -1:
                issue_id = stack_id

        return self.parser.from_json(self.sid_to_dir[stack_id] / f"{stack_id}.json")

    def name(self) -> str:
        return ("rec" if self.frames_field == "frames" else "notrec") + "_loader"


class RequestStackLoader(StackLoader):
    def __init__(self):
        self.stacks: dict[int, Stack] = {}

    def add(self, stacks: list[Stack]):
        self.stacks.update({stack.id: stack for stack in stacks})

    def exists(self, stack_id: int) -> bool:
        return stack_id in self.stacks

    def __call__(self, stack_id: int) -> Stack | None:
        return self.stacks.get(stack_id, None)

    def clear(self):
        self.stacks = {}

    def name(self) -> str:
        raise NotImplementedError
