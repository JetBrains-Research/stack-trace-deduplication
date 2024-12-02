import json
from abc import ABC, abstractmethod
from pathlib import Path

import regex

from ea.sim.main.data.objects.stack import Stack


class StackParser(ABC):
    @classmethod
    @abstractmethod
    def from_dict(cls, x: dict) -> Stack:
        raise NotImplementedError

    @classmethod
    def from_json(cls, path: Path) -> Stack:
        return cls.from_dict(json.loads(path.read_text()))


class MethodNameUnifier:
    number = "(?:0(?:x|\\.)[abcdef\\d]+|[\\d]+)"
    lambda_pattern = rf"(?i)(?<=\$)({number}(?:\/{number})*)"
    generated_pattern = r"(?<=Generated[\w]{0,50}Accessor)([\d]+)"
    proxy_pattern = r"(?<=\$Proxy)([\d]+)"

    @staticmethod
    def unify(name: str) -> str:
        name = regex.sub(MethodNameUnifier.lambda_pattern, "0", name)
        name = regex.sub(MethodNameUnifier.generated_pattern, "0", name)
        name = regex.sub(MethodNameUnifier.proxy_pattern, "0", name)
        return name
