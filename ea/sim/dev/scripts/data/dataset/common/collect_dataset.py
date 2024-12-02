"""
Collecting triplets for training, validating and testing.
"""
import json
import random
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ea.sim.dev.scripts.data.dataset.common.config import DatasetConfig
from ea.sim.dev.scripts.data.dataset.common.objects import Dataset, ReportState


def set_seed(seed: int | None):
    if seed is not None:
        random.seed(seed)


def load_dataset(folder: Path) -> Dataset:
    def load(name: str) -> list[ReportState]:
        path = folder / f"{name}.csv"
        df = pd.read_csv(path)
        return [
            ReportState(report_id, issue_id)
            for (report_id, issue_id) in df.itertuples(index=False)
        ]

    return Dataset(load("train"), load("val"), load("test"))


def save_config(config: DatasetConfig, save_path: Path):
    with save_path.open("w") as file:
        json.dump(asdict(config), file, indent=2)


def save_states(states: list[ReportState], save_path: Path):
    df = pd.DataFrame(data=states, columns=ReportState.columns)
    df.to_csv(save_path, index=False)


def save(folder: Path, dataset: Dataset, config: DatasetConfig):
    save_dir = folder / config.name
    save_dir.mkdir(parents=True, exist_ok=True)

    save_states(dataset.train, save_dir / "train.csv")
    save_states(dataset.val, save_dir / "val.csv")
    save_states(dataset.val, save_dir / "test.csv")
    save_config(config, save_dir / "config.json")
