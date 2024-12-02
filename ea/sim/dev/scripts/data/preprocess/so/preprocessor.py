from pathlib import Path

from loguru import logger

from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import PreprocessStep


class Preprocessor:
    def __init__(self, sources: SlowOpsSources, steps: list[PreprocessStep]):
        self.sources = sources
        self.steps = steps
        self.artifacts = []

    def run(self):
        total = len(self.steps)
        for i, step in enumerate(self.steps, start=1):
            logger.info(f"[{i} / {total}] Starting step '{step.name}'...")
            step.run(self.sources, self.artifacts)
            logger.info(f"[{i} / {total}] Step is finished!")

    def save(self, folder: Path):
        self.sources.save(folder)
        for artifact in self.artifacts:
            artifact.save(folder / "artifacts")
