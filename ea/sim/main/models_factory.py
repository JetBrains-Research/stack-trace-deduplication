from ea.sim.main.configs import SeqCoderConfig
from ea.sim.main.data.stack_loader import StackLoader
from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.methods.classic import *

from ea.sim.main.preprocess.crashes.entry_coders import Crash2Seq
from ea.sim.main.preprocess.entry_coders import Stack2Seq
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.tokenizers import SimpleTokenizer, BPETokenizer
from ea.sim.main.utils import Scope


def create_seq_coder(stack_loader: StackLoader, config: SeqCoderConfig) -> SeqCoder:
    scope = Scope(config.scope_id)
    if scope != Scope.SideProject:
        return SeqCoder(
            stack_loader,
            Stack2Seq(cased=config.cased, sep=config.sep),
            BPETokenizer(sep=config.sep, cased=config.bpe_cased),
            max_len=config.max_len
        )
    elif scope == Scope.SideProject:
        return SeqCoder(
            stack_loader,
            Crash2Seq(cased=False), SimpleTokenizer(), max_len=config.max_len
        )
    else:
        raise ValueError(f"Not found SeqCoder for '{scope.value}' scope")
