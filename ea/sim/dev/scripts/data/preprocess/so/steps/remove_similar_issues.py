
from ea.sim.dev.scripts.data.preprocess.common.artifacs import Artifact
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import PreprocessStep


class RemoveSimilarIssues(PreprocessStep):
    def run(self, sources: SlowOpsSources, artifacts: list[Artifact]):
        raise NotImplementedError

    @property
    def name(self) -> str:
        return "RemoveSimilarIssues"
