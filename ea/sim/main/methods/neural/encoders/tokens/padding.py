import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ea.sim.main.methods.neural.encoders.tokens.base import ListItems
from ea.sim.main.preprocess.id_coder import SpecialTokens

PAD_ID = SpecialTokens.PAD.id


def pad_tokens(tokens: ListItems, tok_len: int, seq_len: int) -> torch.Tensor:
    """
    Takes tokens, pad them in two directions: sequence direction and token direction
    :param tokens: Tokens to pad
    :param tok_len: Length to which each token will be padded
    :param seq_len: Length to which each sequence will be padded
    :return: Tensor with shape (seq_len, tok_len)
    """

    out = [torch.tensor(token.value) for token in tokens[-seq_len:]]
    out = pad_sequence(out, batch_first=True, padding_value=PAD_ID)
    out = F.pad(out, pad=[0, tok_len - out.shape[1], 0, seq_len - out.shape[0]], value=PAD_ID)
    return out
