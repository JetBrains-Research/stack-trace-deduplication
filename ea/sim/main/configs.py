from dataclasses import dataclass

from ea.sim.main.utils import Scope


@dataclass
class SeqCoderConfig:
    scope_id: int
    cased: bool
    bpe_cased: bool
    sep: str
    max_len: int | None

    @staticmethod
    def from_dict(config: dict) -> "SeqCoderConfig":
        return SeqCoderConfig(**config)

    @staticmethod
    def by_scope(scope: Scope) -> "SeqCoderConfig":
        return SeqCoderConfig(scope_id=scope.value, cased=True, bpe_cased=False, sep=".", max_len=None)
