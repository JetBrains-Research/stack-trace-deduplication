from typing import Iterable

from ea.sim.dev.scripts.data.dataset.common.constraints import Constraints
from ea.sim.dev.scripts.data.dataset.common.miners.base import ReportStateMiner
from ea.sim.dev.scripts.data.dataset.common.objects import ReportState
from ea.sim.main.data.buckets.bucket_data import DataSegment, BucketData
from ea.sim.main.data.duplicates import HashStorage


class FinalStateReportMiner(ReportStateMiner):
    def __init__(self, data: BucketData, constraints: Constraints):
        self._data = data
        self._constraints = constraints

    def mine_segment(self, segment: DataSegment) -> Iterable[ReportState]:
        for event in self._data.get_events(segment):
            yield ReportState(event.stack_id, event.issue_id)

    def mine_prev_states(self, segments: list[DataSegment]) -> Iterable[ReportState]:
        # Mines all states in current segments.
        for segment in segments:
            yield from self.mine_segment(segment)

    def mine(self, segment: DataSegment, prev_segments: list[DataSegment] | None = None) -> Iterable[ReportState]:
        hash_storage = HashStorage.get_instance()
        hashes = set()
        if prev_segments is not None:
            prev_states = self.mine_prev_states(prev_segments)
            hashes = set(hash_storage.hash(state.report_id) for state in prev_states)
        cur_states = self.mine_segment(segment)
        for cur_state in cur_states:
            cur_state_hash = hash_storage.hash(cur_state.report_id)
            if cur_state_hash not in hashes:
                hashes.add(cur_state_hash)
                yield cur_state
