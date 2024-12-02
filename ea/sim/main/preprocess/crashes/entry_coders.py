from typing import Any

from ea.sim.main.data.objects.stack import Stack
from ea.sim.main.preprocess.crashes.parsers import LibFrameParser, MethodFrameParser
from ea.sim.main.preprocess.entry_coders import Entry2Seq
from ea.sim.main.preprocess.token import PreTokItem


class Crash2Seq(Entry2Seq):
    def __init__(self, cased: bool = True):
        self.cased = cased
        self.parsers = [
            LibFrameParser(),
            MethodFrameParser()
        ]

        self._args = {"cased": cased}
        self._name = "crash"

    def _parse(self, x: str) -> list[str]:
        entities = [parser.parse(x) for parser in self.parsers]
        return [entity for entity in entities if entity is not None]

    def __call__(self, stack: Stack) -> list[PreTokItem]:
        return [
            PreTokItem(entity, {"has_marker": frame.has_marker})
            for frame in stack.frames for entity in self._parse(frame.name)
        ]

    def state(self) -> dict[str, Any]:
        return self._args

    def name(self) -> str:
        return self._name
