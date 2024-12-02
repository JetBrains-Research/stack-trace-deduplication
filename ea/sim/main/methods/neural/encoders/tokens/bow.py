import torch
from torch import nn

from ea.sim.main.methods.neural.encoders.tokens.base import TokenEncoder, ListItems, BatchListItems
from ea.sim.main.methods.neural.encoders.tokens.padding import pad_tokens
from ea.sim.main.preprocess.id_coder import SpecialTokens
from ea.sim.main.utils import device

PAD_ID = SpecialTokens.PAD.id


class BOWTokenEncoder(TokenEncoder):
    def __init__(self, vocab_size: int, emb_dim: int, dropout: float = 0.0, max_seq_len: int | None = None):
        super().__init__()
        self._emb = nn.Embedding(vocab_size, emb_dim)
        self._emb_dim = emb_dim
        self._dropout = nn.Dropout(dropout)
        self._max_seq_len = max_seq_len

    def encode(self, items: ListItems) -> tuple[torch.Tensor, torch.Tensor]:
        assert all(item.is_split for item in items)
        out = [torch.tensor(item.value).to(device) for item in items]
        out = [self._emb(x).mean(dim=0).view(1, -1) for x in out]
        out = torch.cat(out, dim=0)
        return self._dropout(out)

    def encode_batch(self, items: BatchListItems) -> tuple[torch.Tensor, torch.Tensor]:
        assert all(item.is_split for seq in items for item in seq)
        seq_len = max(len(seq) for seq in items)
        if self._max_seq_len is not None:
            seq_len = min(seq_len, self._max_seq_len)
        max_tok_len = max(len(token) for seq in items for token in seq)
        indexes = torch.cat(
            [pad_tokens(item, max_tok_len, seq_len).unsqueeze(0) for item in items],
            dim=0
        ).to(device)  # (batch_size, seq_len, sub_tokens_cnt)

        embs = self._emb(indexes)  # (batch_size, seq_len, sub_tokens_cnt, emb_dim)
        padding_mask = torch.eq(indexes, PAD_ID)  # (batch_size, seq_len, sub_tokens_cnt)
        embs_masked = embs.masked_fill(padding_mask.unsqueeze(-1), 0)
        sum_embeddings = embs_masked.sum(dim=2)
        num_non_padding = (~padding_mask).sum(dim=2, keepdim=True)
        mean_embeddings = sum_embeddings / num_non_padding.clamp(min=1)
        mean_embeddings = self._dropout(mean_embeddings)
        return mean_embeddings, padding_mask

    @property
    def dim(self) -> int:
        return self._emb_dim
