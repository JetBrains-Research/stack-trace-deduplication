import torch
import torch.nn as nn

from ea.sim.main.utils import StackId
from ea.sim.main.methods.retrieval_model import RetrievalModel

class S3MMockRetrievalModel(RetrievalModel):
    "Mocks the bechaviour of S3M assuming that embeddings are saved in cache. Used to measure time of retrieval."

    def __init__(self, dim_size, top_n=10):
        self.dim_size = dim_size
        self._top_n = top_n

        self.head = nn.Sequential(
            nn.Linear(dim_size * 2 + 1, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, 1),
        )

    def get_embedding(self, stack_ids: list[StackId]) -> torch.Tensor:
        # returns random embedding
        return torch.rand(len(stack_ids), self.dim_size)

    def search(self, anchor_id: StackId, filter_ids: list[StackId] | None = None) -> list[StackId]:
        batch_size = 64

        anchor_embedding = self.get_embedding([anchor_id])

        assert filter_ids is not None

        similarity_scores = []

        for i in range(0, len(filter_ids), batch_size):
            batch_ids = filter_ids[i:i + batch_size]
            
            batch_embedding = self.get_embedding(batch_ids)

            # calculate similarity
            input_embs = anchor_embedding.repeat(len(batch_ids), 1)

            input_features = torch.cat([(input_embs - batch_embedding).norm(dim=1).reshape(-1, 1), (input_embs * batch_embedding), (input_embs + batch_embedding) / 2], dim=1)

            scores = self.head(input_features).squeeze()

            if len(scores.shape) == 0:
                continue
            if len(scores) == 1:
                similarity_scores.append(scores.cpu().detach().item())
            else:
                similarity_scores.extend(scores.cpu().detach().tolist())

        sorted_ids = [x for _, x in sorted(zip(similarity_scores, filter_ids), reverse=True)]

        return sorted_ids
