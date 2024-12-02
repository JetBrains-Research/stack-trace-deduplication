"""
Mining reports for training.
"""

from abc import ABC, abstractmethod
from typing import Iterable

from ea.sim.dev.scripts.data.dataset.common.objects import ReportState
from ea.sim.main.data.buckets.bucket_data import DataSegment


class ReportStateMiner(ABC):
    @abstractmethod
    def mine(self, segment: DataSegment, prev_segments: list[DataSegment] | None = None) -> Iterable[ReportState]:
        raise NotImplementedError
