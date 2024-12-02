import json
from abc import ABC, abstractmethod
from pathlib import Path

from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.utils import StackId, Score


class ScoringModel(ABC):
    @abstractmethod
    def predict(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[Score]:
        raise NotImplementedError

    @property
    @abstractmethod
    def min_score(self) -> float:
        raise NotImplementedError


class SimpleScoringModel(ScoringModel):
    def __init__(self, model: SimStackModel, top_n: int | None = None):
        self._model = model
        self._top_n = top_n

    def predict(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[Score]:
        candidate_ids = candidate_ids[:self._top_n]
        return self._model.predict(anchor_id, candidate_ids)

    @property
    def min_score(self) -> float:
        return self._model.min_score


class CachedScoringModel(ScoringModel):
    file_name: str = "scoring_cache.json"

    def __init__(self, model: ScoringModel):
        self._model = model
        self._cache = []

    def predict(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[Score]:
        candidates = candidate_ids
        scores = self._model.predict(anchor_id, candidates)
        self._cache.append({
            "anchor_id": anchor_id,
            "candidates": candidates,
            "scores": scores
        })
        return scores

    def save_cache(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / self.file_name).write_text(json.dumps(self._cache, indent=2))

    @property
    def min_score(self) -> float:
        return self._model.min_score
