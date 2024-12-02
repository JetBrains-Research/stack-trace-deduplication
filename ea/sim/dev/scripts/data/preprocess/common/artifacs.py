import json
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    data: Any
    name: str

    @abstractmethod
    def save(self, folder: Path):
        raise NotImplementedError


class JsonArtifact(Artifact):
    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.name}.json").write_text(json.dumps(self.data, indent=2))


class CsvArtifact(Artifact):
    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        self.data.to_csv(folder / f"{self.name}.csv", index=False)
