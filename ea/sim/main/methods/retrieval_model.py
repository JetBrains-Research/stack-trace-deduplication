import json
from abc import abstractmethod
from pathlib import Path

from ea.sim.main.methods.index import Index
from ea.sim.main.utils import StackId


class RetrievalModel:
    @abstractmethod
    def search(self, anchor_id: StackId, filter_ids: list[StackId] | None = None) -> list[StackId]:
        raise NotImplementedError


class IndexRetrievalModel(RetrievalModel):
    def __init__(self, index: Index, top_n: int):
        self._top_n = top_n
        self._index = index

    def search(self, anchor_id: StackId, filter_ids: list[StackId] | None = None) -> list[StackId]:
        return self._index.search(anchor_id, self._top_n, filter_ids)


class CachedRetrievalModel(RetrievalModel):
    _file_name = "retrieval_cache.json"

    def __init__(self, model: RetrievalModel):
        super().__init__()
        self._model = model
        self._cache = {}
        self._top_n = self._model._top_n

    def search(self, anchor_id: StackId, filter_ids: list[StackId] | None = None) -> list[StackId]:
        if anchor_id not in self._cache:
            candidates = self._model.search(anchor_id, filter_ids)
            self._cache[anchor_id] = candidates
        return self._cache[anchor_id][:self._top_n]

    def save_cache(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / self._file_name
        file_path.write_text(json.dumps(self._cache, indent=2))

    def load_cache(self, folder: Path):
        file_path = folder / self._file_name
        json_cache = json.loads(file_path.read_text())
        self._cache = {int(k): v for k, v in json_cache.items()}


class DummyRetrievalModel(RetrievalModel):
    """
    Retrieve all stacks. Is used for evaluation of only-cross-encoder pipelines such 'Lerch', 'FaST' etc.
    """
    def __init__(self):
        self._top_n = int(1e9)

    def search(self, anchor_id: StackId, filter_ids: list[StackId] | None = None) -> list[StackId]:
        return filter_ids
    