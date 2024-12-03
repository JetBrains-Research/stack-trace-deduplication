import json
import re
from typing import Any, Iterable

from loguru import logger
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer, Trainer

from ea.sim.main.preprocess.id_coder import SpecialTokens
from ea.sim.main.preprocess.token import PreTokItem, PostTokItem
from ea.sim.main.preprocess.tokenizers import Tokenizer
from ea.sim.main.utils import ARTIFACTS_DIR

SAVE_FOLDER = ARTIFACTS_DIR / "bpe"
TEXT_PATH = SAVE_FOLDER / "text.txt"
VOCAB_PATH = SAVE_FOLDER / "vocab.json"


class CamelCaseAndDotPreTokenizer:
    reg_exp_1 = "([A-Z]+)"
    reg_exp_2 = "([A-Z][a-z]+)"

    def __init__(self, sep: str, cased: bool):
        self._sep = sep
        self._cased = cased

    def tokenize_camel_case(self, word: str) -> list[str]:
        # https://stackoverflow.com/a/37697078
        word = re.sub(CamelCaseAndDotPreTokenizer.reg_exp_1, r" \1", word)
        word = re.sub(CamelCaseAndDotPreTokenizer.reg_exp_2, r" \1", word)
        return word.split()

    def pre_tokenize(self, pretok: str) -> list[str]:
        tokens = pretok.split(self._sep)
        tokens = [sub_token for token in tokens for sub_token in self.tokenize_camel_case(token)]
        if not self._cased:
            tokens = [token.lower() for token in tokens]
        return tokens


class BPETokenizer(Tokenizer):
    special_tokens = [token.value for token in SpecialTokens.all]

    def __init__(
            self,
            refit: bool = False,
            vocab_size: int | None = 10_000, min_freq: int | None = 1,
            sep: str = ".", cased: bool = False,
            source_type: str = "file",  # or "iterator"
    ):
        super().__init__()
        self._refit = refit
        self._trainer_args = {
            "vocab_size": vocab_size,
            "min_freq": min_freq,
            "special_tokens": BPETokenizer.special_tokens,
        }
        self._source_type = source_type
        self._cased = cased

        # HuggingFace custom PreTokenizer is not documented enough :(
        self._custom_pre_tokenizer = CamelCaseAndDotPreTokenizer(sep=sep, cased=cased)
        self._tokenizer = HFTokenizer(BPE(special_tokens=[SpecialTokens.UNK.value]))
        self._tokenizer.pre_tokenizer = Whitespace()

    def fit_from_file(self, texts: Iterable[list[PreTokItem]], trainer: Trainer) -> "BPETokenizer":
        logger.debug(f"BPE Tokenizer fitting from file '{TEXT_PATH}'")
        with TEXT_PATH.open("w") as file:
            for text in texts:
                for pre_tok in text:
                    # And use `Whitespace` PreTokenizer.
                    token = " ".join(self._custom_pre_tokenizer.pre_tokenize(pre_tok.value))
                    file.write(f"{token}\n")
        self._tokenizer.train([str(TEXT_PATH)], trainer=trainer)
        return self

    def fit_from_iterator(self, texts: Iterable[list[PreTokItem]], trainer: Trainer) -> "BPETokenizer":
        logger.debug("BPE Tokenizer fitting from iterator")
        # And use `WhiteSpace` PreTokenizer.
        data = (
            " ".join(self._custom_pre_tokenizer.pre_tokenize(pre_tok.value))
            for text in texts for pre_tok in text
        )
        self._tokenizer.train_from_iterator(data, trainer=trainer)
        return self

    def fit(self, texts: Iterable[list[PreTokItem]]) -> "BPETokenizer":
        is_training_required = (not VOCAB_PATH.exists()) or self._refit
        if not is_training_required:
            # Loading.
            state = json.loads(VOCAB_PATH.read_text())
            self.load_state(state)
            logger.debug(f"BPE Tokenizer loaded from state file '{VOCAB_PATH}'")
            return self

        # Fitting from scratch.
        SAVE_FOLDER.mkdir(exist_ok=True, parents=True)
        trainer = BpeTrainer(**self._trainer_args)
        if self._source_type == "file":
            self.fit_from_file(texts, trainer)
        elif self._source_type == "iterator":
            self.fit_from_iterator(texts, trainer)
        else:
            raise ValueError(f"Unknown source type: {self._source_type}")

        VOCAB_PATH.write_text(json.dumps(self.state(), indent=2))
        logger.debug(f"BPE Tokenizer fitted, saved to '{VOCAB_PATH}'")
        return self

    def partial_fit(self, texts: Iterable[list[PreTokItem]]) -> "BPETokenizer":
        logger.debug("Partial fitting is not implement for BPETokenizer, full fitting")
        return self.fit(texts)

    def encode(self, text: list[PreTokItem]) -> list[PostTokItem[int]]:
        return [
            PostTokItem(
                value=self._tokenizer.encode(item.value if self._cased else item.value.lower()).ids,
                extras=item.extras
            )
            for item in text
        ]

    def split(self, text: list[PreTokItem]) -> list[PostTokItem[str]]:
        return [
            PostTokItem(
                value=self._tokenizer.encode(item.value).tokens,
                extras=item.extras
            )
            for item in text
        ]

    def to_str(self, id: int) -> str:
        return self._tokenizer.id_to_token(id)

    def __len__(self) -> int:
        return self._tokenizer.get_vocab_size()

    def state(self) -> dict[str, Any]:
        return {"coder": json.loads(self._tokenizer.to_str(pretty=True))}

    def load_state(self, state: dict):
        self._tokenizer = HFTokenizer.from_str(json.dumps(state["coder"]))

    def name(self) -> str:
        return "bpe"

    @property
    def token_split(self) -> bool:
        return True
