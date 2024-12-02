import json
from pathlib import Path

import numpy as np
import torch

from ea.sim.main.methods.neural.neural import NeuralModel
from ea.sim.main.methods.neural.similarity import Similarity
from ea.sim.main.utils import StackId


class CacheModel(NeuralModel):
    ids_file_name: str = "report_ids.json"
    embeddings_file_name: str = "embeddings.npz"

    def __init__(self, folder: Path, similarity: Similarity, trim: int | None = None):
        super().__init__()

        report_ids = json.loads((folder / CacheModel.ids_file_name).read_text())
        embeddings = np.load(folder / CacheModel.embeddings_file_name)["embeddings"]

        if trim is not None:
            embeddings = embeddings[:, :trim]
            norm = np.linalg.norm(embeddings, 2, axis=1, keepdims=True)
            embeddings = np.where(norm == 0, embeddings, embeddings / norm)

        embeddings = embeddings.astype("float32")
        self._embeddings = {report_id: embedding for report_id, embedding in zip(report_ids, embeddings)}
        self._out_dim = embeddings.shape[1]
        self.similarity = similarity
        self.similarity.eval()

    @torch.no_grad()
    def encode(self, stack_ids: list[StackId]) -> torch.Tensor:
        embeddings = [self._embeddings[stack_id] for stack_id in stack_ids]
        embeddings = np.vstack(embeddings)
        return torch.tensor(embeddings)

    @torch.no_grad()
    def predict(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[float]:
        anchor = self.encode([anchor_id])
        candidates = self.encode(candidate_ids)
        return self.similarity(anchor, candidates).detach().cpu().tolist()

    def out_dim(self) -> int:
        return self._out_dim

    def name(self) -> str:
        raise NotImplementedError

    def save(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path, *, device: torch.device | str):
        raise NotImplementedError

    @property
    def min_score(self) -> float:
        return float(self.similarity.min_value)
