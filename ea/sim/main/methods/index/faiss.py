import faiss as faiss_lib
import numpy as np
from loguru import logger

from ea.sim.main.methods.index import Index
from ea.sim.main.methods.neural.encoder import Encoder
from ea.sim.main.utils import StackId


class FAISS(Index):
    """
    FAISS index with cosine metric.
    """

    def __init__(self, encoder: Encoder):
        self._encoder = encoder

        self._index = None
        self._sup_stack_ids = []  # used for building index
        self._index_to_stack_id: dict[int, StackId] = {}
        self._stack_id_to_index: dict[StackId, int] = {}

    def encode(self, stack_ids: list[StackId]) -> np.ndarray:
        vectors = self._encoder.encode(stack_ids).detach().cpu().numpy()
        faiss_lib.normalize_L2(vectors)
        return vectors.astype(np.float32)

    def fit(self, stack_ids: list[StackId]) -> "FAISS":
        if self._index is not None:
            logger.debug("Index model is already fitted, call skipped")
            return self

        self._index = faiss_lib.IndexFlatIP(self._encoder.out_dim())
        self._sup_stack_ids = stack_ids.copy()
        self._index_to_stack_id = {i: stack_id for i, stack_id in enumerate(stack_ids)}
        self._stack_id_to_index = {stack_id: i for i, stack_id in self._index_to_stack_id.items()}
        self._index.add(self.encode(stack_ids))
        return self

    def insert(self, stack_ids: list[StackId]) -> "Index":
        stack_ids = [stack_id for stack_id in stack_ids if stack_id not in self._stack_id_to_index]

        if len(stack_ids) == 0:
            return self

        self._index.add(self.encode(stack_ids))
        total_stacks = len(self._stack_id_to_index)
        self._stack_id_to_index.update({stack_id: i + total_stacks for i, stack_id in enumerate(stack_ids)})
        self._index_to_stack_id.update({i + total_stacks: stack_id for i, stack_id in enumerate(stack_ids)})

        assert self._index.ntotal == len(self._stack_id_to_index)

    def refit(self) -> "FAISS":
        sup_stack_ids = self._sup_stack_ids

        self._index = None
        self._sup_stack_ids = []
        self._stack_id_to_index = {}
        self._index_to_stack_id = {}

        return self.fit(sup_stack_ids)

    def search(self, anchor_id: int, k: int, filter_ids: list[StackId] | None = None) -> list[int]:
        assert self._index.ntotal == len(self._stack_id_to_index)
        query = self.encode([anchor_id]).reshape(1, -1)
        if filter_ids is None:
            _, indexes = self._index.search(query, k=k)
        else:
            filter_ids = [
                self._stack_id_to_index[stack_id]
                for stack_id in filter_ids if stack_id in self._stack_id_to_index
            ]
            selector = faiss_lib.IDSelectorArray(filter_ids)
            params = faiss_lib.SearchParameters(sel=selector)
            _, indexes = self._index.search(query, k=k, params=params)

        return [self._index_to_stack_id[index] for index in indexes.tolist()[0] if index != -1]

    def reset(self):
        self._index = None
        self._sup_stack_ids = []
        self._stack_id_to_index = {}
        self._index_to_stack_id = {}
