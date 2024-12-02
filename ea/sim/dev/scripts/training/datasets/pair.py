import random
from pathlib import Path
from typing import NamedTuple
from typing import TypeAlias

from loguru import logger
from torch.utils.data import Dataset

from ea.sim.dev.scripts.training.datasets.common import load_state, State, load_groups, SamplingTechnique
from ea.sim.main.methods.neural.encoders.objects import ItemProcessor, Item
from ea.sim.main.preprocess.seq_coder import SeqCoder

Anchor: TypeAlias = Item
Doc: TypeAlias = Item


class Pair(NamedTuple):
    query_id: int
    doc_id: int


def create_pairs(state: State, max_per_group: int | None = None) -> list[Pair]:
    groups = load_groups(state)
    pairs = []
    for reports in groups.values():
        # Form groups (report_id_1, report_id_2), (report_id_3, report_id_4), (report_id_5, report_id_6)...
        if len(reports) <= 1:
            continue
        if max_per_group is not None:
            reports = reports[:max_per_group * 2]
        else:
            reports = reports[:len(reports) // 2 * 2]
        group_pairs = [Pair(query_id, doc_id) for query_id, doc_id in zip(reports[::2], reports[1::2])]
        pairs.extend(group_pairs)
    return pairs


def create_pairs_all_with_all(state: State, max_pairs_per_group: int | None = None) -> list[Pair]:
    groups = load_groups(state)
    pairs = []
    for reports in groups.values():
        num_pairs_in_group = (len(reports) * (len(reports) - 1)) // 2
        if max_pairs_per_group is None or num_pairs_in_group <= max_pairs_per_group:
            pairs.extend([
                Pair(reports[i], reports[j])
                for i in range(len(reports)) for j in range(i + 1, len(reports))
            ])
        else:
            all_pairs_indexes = [(i, j) for i in range(len(reports)) for j in range(i + 1, len(reports))]
 
            sampled = random.sample(all_pairs_indexes, max_pairs_per_group)
 
            pairs.extend([
                Pair(reports[i], reports[j])
                for i, j in sampled
            ])
    return pairs


class PairDataset(Dataset):
    def __init__(
            self,
            file_path: Path,
            seq_coder: SeqCoder,
            max_per_group: int | None = None,
            random_size: int | None = None,
            sampling_technique: SamplingTechnique = SamplingTechnique.ALL_WITH_ALL
    ):
        super().__init__()
        self._pairs = self.init_pairs(file_path, max_per_group, random_size, sampling_technique)
        self._item_processor = ItemProcessor(seq_coder)

    @staticmethod
    def init_pairs(
            file_path: Path,
            max_per_group: int | None = None,
            random_size: int | None = None,
            sampling_technique: SamplingTechnique = SamplingTechnique.ALL_WITH_ALL
    ) -> list[Pair]:
        state = load_state(file_path)
        logger.debug(f"Loaded state with size {len(state)} from '{file_path}'")
        if sampling_technique == SamplingTechnique.ALL_WITH_ALL:
            pairs = create_pairs_all_with_all(state, max_per_group)
        elif sampling_technique == SamplingTechnique.LINEAR:
            pairs = create_pairs(state, max_per_group)
        else:
            raise ValueError(f"Unknown sampling technique: {sampling_technique}")
        logger.debug(f"Formed {len(pairs)} query-doc pairs")
        if random_size is not None:
            pairs = sorted(pairs, key=lambda k: random.random())
            pairs = pairs[:random_size]
        return pairs

    def __getitem__(self, item: int) -> tuple[Anchor, Doc]:
        pair = self._pairs[item]
        anchor = self._item_processor(pair.query_id)
        doc = self._item_processor(pair.doc_id)
        return anchor, doc

    def __len__(self) -> int:
        return len(self._pairs)

    @staticmethod
    def collate_fn(batch: list[tuple[Item, ...]]) -> list[tuple[Item, ...]]:
        return batch
