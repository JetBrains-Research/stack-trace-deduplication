from pathlib import Path

import torch
from tqdm.auto import tqdm

from ea.sim.main.methods.neural.neural import NeuralModel
from ea.sim.main.methods.neural.encoders.objects import ItemProcessor
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.methods.neural.similarity import Similarity
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.utils import StackId, device



class SiamMultiModalModel(NeuralModel):
    BATCH_SIZE: int = 8

    def __init__(self, seq_coder: SeqCoder, encoder: Encoder, similarity: Similarity, verbose: bool = False):
        super().__init__()
        self._item_processor = ItemProcessor(seq_coder)
        self.encoder = encoder
        self.similarity = similarity
        self.verbose = verbose
        self.embedding_cache: dict[StackId, torch.Tensor] = {}

    @torch.no_grad()
    def encode(self, stack_ids: list[StackId]) -> torch.Tensor:
        embs = []
        start_indexes_gen = range(0, len(stack_ids), self.BATCH_SIZE)
        start_indexes_gen = tqdm(
            start_indexes_gen,
            desc="Encoding stacks",
            total=len(stack_ids) // self.BATCH_SIZE,
            disable=not self.verbose
        )
        for start_idx in start_indexes_gen:
            items = [self._item_processor(stack_id) for stack_id in stack_ids[start_idx:start_idx + self.BATCH_SIZE]]
            embs.append(self.encoder(items).detach().cpu())
        return torch.cat(embs, dim=0)

    def cached_encode(self, stack_ids: list[StackId]) -> torch.Tensor:
        stack_ids_without_cache = [stack_id for stack_id in stack_ids if stack_id not in self.embedding_cache]
        if stack_ids_without_cache:
            embs = self.encode(stack_ids_without_cache)
            for stack_id, emb in zip(stack_ids_without_cache, embs):
                self.embedding_cache[stack_id] = emb.reshape(1, -1)
        return torch.cat([self.embedding_cache[stack_id] for stack_id in stack_ids], dim=0)

    @torch.no_grad()
    def predict(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[float]:
        predict_batch_size = 128
        scores = []
        anchor_emb = self.cached_encode([anchor_id])
        similarity = self.similarity.cpu()

        for start_idx in range(0, len(candidate_ids), predict_batch_size):
            candidate_batch = candidate_ids[start_idx:start_idx + predict_batch_size]
            candidate_embs = self.cached_encode(candidate_batch)

            scores.extend(similarity(anchor_emb, candidate_embs).detach().cpu().tolist())

        return scores

    def out_dim(self) -> int:
        return self.encoder.out_dim

    def name(self) -> str:
        raise NotImplementedError

    def save(self, folder: Path):
        raise NotImplementedError

    def load(self, folder: Path, *, device: torch.device | str):
        raise NotImplementedError

    @property
    def min_score(self) -> float:
        return float(self.similarity.min_value)
