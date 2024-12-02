import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ea.sim.main.methods.neural.encoders.tokens.base import TokenEncoder, ListItems, BatchListItems
from ea.sim.main.methods.neural.encoders.tokens.padding import pad_tokens
from ea.sim.main.preprocess.id_coder import SpecialTokens
from ea.sim.main.utils import device

PAD_ID = SpecialTokens.PAD.id


class RNNTokenEncoder(TokenEncoder):

    def __init__(self, vocab_size: int, emb_dim: int, dropout: float = 0.0, hidden_size: int = 50, max_seq_len: int | None = None):
        super().__init__()
        self._emb = nn.Embedding(vocab_size, emb_dim)
        self._emb_dim = emb_dim
        self._dropout = nn.Dropout(dropout)
        self._max_seq_len = max_seq_len
        self._hidden_size = hidden_size
        self._rnn = nn.LSTM(input_size=emb_dim, hidden_size=self._hidden_size, num_layers=1, batch_first=True, bidirectional=True)

    def encode(self, items: ListItems) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
        batch_size = embs.shape[0]
        padding_mask = torch.eq(indexes, PAD_ID)  # (batch_size, seq_len, sub_tokens_cnt)

        lens = max_tok_len - padding_mask.sum(dim=2).reshape(-1).cpu()

        inds_not_empty = torch.where(lens > 0)[0]
        tensors = pack_padded_sequence(embs.reshape(-1, max_tok_len, self._emb_dim)[inds_not_empty],
                                       lens[inds_not_empty],
                                       batch_first=True,
                                       enforce_sorted=False)
        output, (_, _) = self._rnn(tensors)

        output, _ = pad_packed_sequence(output, batch_first=True, padding_value=0.0, total_length=max_tok_len)

        # output: (len(inds_not_empty), max_tok_len, hidden_size * 2)

        output = output.sum(dim=1)  # (len(inds_not_empty), hidden_size * 2)

        embeddings = torch.zeros((batch_size * seq_len, self._hidden_size * 2)).to(device)

        embeddings[inds_not_empty] = output

        embeddings = embeddings.reshape(batch_size, seq_len, self._hidden_size * 2)

        return embeddings, padding_mask



    @property
    def dim(self) -> int:
        return self._hidden_size * 2
