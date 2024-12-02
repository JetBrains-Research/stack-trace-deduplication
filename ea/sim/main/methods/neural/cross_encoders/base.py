from abc import abstractmethod, ABC

import torch
from torch import nn

from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.encoders.objects import ItemProcessor
from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.preprocess.seq_coder import SeqCoder

class CrossEncoder(nn.Module, ABC):
    @abstractmethod
    def forward(self, first: list[Item], second: list[Item]) -> torch.Tensor:
        """
        Encodes items into vectors.
        :param first: Sequence of N items
        :param second: Sequence of N items
        :return: Tensor of size (N, ) with similarity scores
        """
        raise NotImplementedError
    
class SimStackModelFromCrossEncoder(SimStackModel):
    BATCH_SIZE: int = 32

    def __init__(self, cross_encoder: CrossEncoder, seq_coder: SeqCoder):
        self.cross_encoder = cross_encoder
        self._item_processor = ItemProcessor(seq_coder)

    def pairs_similarities(self, first: list[Item], second: list[Item]) -> list[float]:
        return self.cross_encoder(first, second).detach().cpu().tolist()

    def predict(self, anchor_id: int, stack_ids: list[int]) -> list[float]:
        anchor = self._item_processor(anchor_id)
        candidates = [self._item_processor(stack_id) for stack_id in stack_ids]
        
        similarities = []
        start_indexes_gen = range(0, len(candidates), self.BATCH_SIZE)
        for start_idx in start_indexes_gen:
            batch_candidates = candidates[start_idx:start_idx + self.BATCH_SIZE]
            similarities.extend(self.pairs_similarities([anchor] * len(batch_candidates), batch_candidates))
        
        return similarities

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]] | None = None,
            unsup_data: list[int] | None = None
    ) -> 'SimStackModel':
        return self
    
    def name(self) -> str:
        return "CrossEncoderStackModel"
    
    @property
    def min_score(self) -> float:
        return -float("inf")