import torch
from torch import nn
from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm

from loguru import logger

from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.methods.neural.encoders.tokens.base import TokenEncoder, ListItems, BatchListItems
from ea.sim.main.methods.neural.encoders.tokens.padding import pad_tokens
from ea.sim.main.preprocess.id_coder import SpecialTokens
from ea.sim.main.utils import ARTIFACTS_DIR, device
from ea.sim.main.utils import StackId

PAD_ID = SpecialTokens.PAD.id

class SkipGramBOW(TokenEncoder):

    _save_folder = ARTIFACTS_DIR / "skip_gram"

    @classmethod
    def initialize(cls, stack_ids: list[StackId], seq_coder: SeqCoder, vocab_size: int = 10000, embedding_dim: int = 50, window: int = 3):
        """
        Trains a skip-gram model on the given stack ids. saves embeddings to _save_folder as torch tensor.
        """
        logger.info(f"Training skip-gram model for {len(stack_ids)} stacks")
        all_data = (list(map(str, token.all_ids)) for stack_id in stack_ids for token in seq_coder(stack_id))
        all_data = tqdm(all_data, desc="Skip-gram training")

        model = Word2Vec(all_data, vector_size=embedding_dim, window=window, min_count=1, workers=4, sg=1)

        logger.info("Skip-gram model trained")

        embeddings = torch.zeros((vocab_size, embedding_dim))

        for i in range(vocab_size):
            if str(i) in model.wv:
                embeddings[i] = torch.tensor(model.wv[str(i)])
        
        logger.info(f"Saving skip-gram embeddings to {cls._save_folder}")

        cls._save_folder.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cls._save_folder / f"embeddings_{window}.pt")
    
    def __init__(self, embedding_dim: int = 50, dropout: float = 0.0, max_seq_len: int | None = None, freeze: bool = True, window: int = 3):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._embeddings = torch.load(self._save_folder / f"embeddings_{window}.pt")
        self._embeddings = self._embeddings.to(device)
        self._embeddings = nn.Embedding.from_pretrained(self._embeddings, freeze=freeze)
        self._dropout = nn.Dropout(dropout)
        self._max_seq_len = max_seq_len

    def encode(self, items: ListItems) -> tuple[torch.Tensor, torch.Tensor]:
        assert all(item.is_split for item in items)
        out = [torch.tensor(item.value).to(device) for item in items]
        out = [self._embeddings(x).mean(dim=0).view(1, -1) for x in out]
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

        embs = self._embeddings(indexes)  # (batch_size, seq_len, sub_tokens_cnt, emb_dim)
        padding_mask = torch.eq(indexes, PAD_ID)  # (batch_size, seq_len, sub_tokens_cnt)
        embs_masked = embs.masked_fill(padding_mask.unsqueeze(-1), 0)
        sum_embeddings = embs_masked.sum(dim=2)
        num_non_padding = (~padding_mask).sum(dim=2, keepdim=True)
        mean_embeddings = sum_embeddings / num_non_padding.clamp(min=1)
        mean_embeddings = self._dropout(mean_embeddings)
        return mean_embeddings, padding_mask

    @property
    def dim(self) -> int:
        return self._embedding_dim
