from abc import abstractmethod

from ea.sim.dev.scripts.data.preprocess.common.artifacs import Artifact
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources


class PreprocessStep:
    @abstractmethod
    def run(self, sources: SlowOpsSources, artifacts: list[Artifact]):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
