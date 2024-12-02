from dataclasses import dataclass
from pathlib import Path

from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.utils import Scope


@dataclass
class Paths:
    reports_dir: Path
    state_path: Path
    save_dir: Path
    actions_path: Path | None = None
    constraints_path: Path | None = None


@dataclass
class SegmentsConfig:
    train: DataSegment
    val: DataSegment
    test: DataSegment


@dataclass
class DatasetConfig:
    name: str
    random_seed: int | None
    segments: SegmentsConfig
    scope: Scope
