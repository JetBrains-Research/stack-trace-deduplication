import random
from pathlib import Path
from typing import NamedTuple, TypeAlias

from loguru import logger
from torch.utils.data import Dataset

from ea.sim.dev.scripts.training.datasets.common import load_state, State, load_groups, SamplingTechnique
from ea.sim.main.methods.neural.encoders.objects import Item, ItemProcessor
from ea.sim.main.preprocess.seq_coder import SeqCoder

Anchor: TypeAlias = Item
Positive: TypeAlias = Item
Negative: TypeAlias = Item


class Triplet(NamedTuple):
    query_id: int
    positive_id: int
    negative_id: int


def create_triplets(state: State, max_per_group: int | None = None) -> list[Triplet]:
    all_report_ids = state.report_ids
    groups = load_groups(state)
    triplets = []
    for reports in groups.values():
        if len(reports) <= 1:
            continue
        if max_per_group is not None:
            reports = reports[:max_per_group * 2]
        else:
            reports = reports[:len(reports) // 2 * 2]
        group_triplets = [
            Triplet(query_id, positive_id, random.choice(all_report_ids))
            for query_id, positive_id in zip(reports[::2], reports[1::2])
        ]
        triplets.extend(group_triplets)
    return triplets


def create_triplets_all_with_all(state: State, max_pairs_per_group: int | None = None) -> list[Triplet]:
    all_report_ids = state.report_ids
    groups = load_groups(state)
    pairs = []
    for reports in groups.values():
        num_pairs_in_group = (len(reports) * (len(reports) - 1)) // 2
        if max_pairs_per_group is None or num_pairs_in_group <= max_pairs_per_group:
            pairs.extend([
                Triplet(reports[i], reports[j], random.choice(all_report_ids))
                for i in range(len(reports)) for j in range(i + 1, len(reports))
            ])
        else:
            all_pairs_indexes = [(i, j) for i in range(len(reports)) for j in range(i + 1, len(reports))]
 
            sampled = random.sample(all_pairs_indexes, max_pairs_per_group)
 
            pairs.extend([
                Triplet(reports[i], reports[j], random.choice(all_report_ids))
                for i, j in sampled
            ])
    return pairs


class TripletDataset(Dataset):
    def __init__(
            self,
            file_path: Path,
            seq_coder: SeqCoder,
            max_per_group: int | None = None,
            random_size: int | None = None,
            sampling_technique: SamplingTechnique = SamplingTechnique.ALL_WITH_ALL
    ):
        self._triplets = self.init_triplets(file_path, max_per_group, random_size, sampling_technique)
        self._item_processor = ItemProcessor(seq_coder)

    @staticmethod
    def init_triplets(file_path: Path, max_per_group: int | None = None, random_size: int | None = None,
                      sampling_technique: SamplingTechnique = SamplingTechnique.ALL_WITH_ALL) -> list[Triplet]:
        state = load_state(file_path)
        logger.debug(f"Loaded state with size {len(state)} from '{file_path}'")
        if sampling_technique == SamplingTechnique.ALL_WITH_ALL:
            triplets = create_triplets_all_with_all(state, max_per_group)
        elif sampling_technique == SamplingTechnique.LINEAR:
            triplets = create_triplets(state, max_per_group)
        else:
            raise ValueError(f"Unknown sampling technique: {sampling_technique}")
        logger.debug(f"Formed {len(triplets)} query-positive-negative triplets")
        if random_size is not None:
            triplets = sorted(triplets, key=lambda k: random.random())
            triplets = triplets[:random_size]
        return triplets

    def __getitem__(self, item: int) -> tuple[Anchor, Positive, Negative]:
        triplet = self._triplets[item]
        anchor = self._item_processor(triplet.query_id)
        positive = self._item_processor(triplet.positive_id)
        negative = self._item_processor(triplet.negative_id)
        return anchor, positive, negative

    def __len__(self) -> int:
        return len(self._triplets)

    @staticmethod
    def collate_fn(batch: list[tuple[Item, ...]]) -> list[tuple[Item, ...]]:
        return batch
