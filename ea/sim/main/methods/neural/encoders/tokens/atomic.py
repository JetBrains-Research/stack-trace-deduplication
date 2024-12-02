import torch
from torch import nn

from ea.sim.main.methods.neural.encoders.tokens.base import TokenEncoder, ListItems, BatchListItems
from ea.sim.main.utils import device


class AtomicTokenEncoder(TokenEncoder):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self._emb = nn.Embedding(vocab_size, emb_dim)
        self._emb_dim = emb_dim
        self._dropout = nn.Dropout(dropout)

    def encode(self, items: ListItems) -> torch.Tensor | tuple[torch.Tensor, ...]:
        assert all(not item.is_split for item in items)
        out = torch.tensor([item.value for item in items]).to(device)
        out = self._emb(out)
        return self._dropout(out)

    def encode_batch(self, items: BatchListItems) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return self._emb_dim
