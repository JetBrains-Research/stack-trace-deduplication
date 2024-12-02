import typing as tp

from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.utils import StackId, Timestamp


class StackEvent(tp.NamedTuple):
    stack_id: int
    ts: int
    label: int


def calculate_unique_stacks(stack_ids: list[StackId]) -> list[StackId]:
    hash_storage = HashStorage.get_instance()
    hashes = set()
    unique_stack_ids = []
    for stack_id in sorted(stack_ids):
        stack_hash = hash_storage.hash(stack_id)
        if stack_hash not in hashes:
            hashes.add(stack_hash)
            unique_stack_ids.append(stack_id)
    return unique_stack_ids


class Issue:
    def __init__(self, id: int, ts: int):
        self.id = id
        self.stack_events: dict[int, StackEvent] = {}
        self._last_update_tss: list[int] = [ts]
        self._all_unique_stack_ids: list[int] | None = None  # None if not eval yet

    def _recalculate_unique_stacks(self):
        # Returns only unique stacks from issue based on hash.
        hash_storage = HashStorage.get_instance()
        self._unique_stack_ids = []
        unique_stack_hashes = set()

        for stack_id in sorted(self.stack_ids()):
            stack_hash = hash_storage.hash(stack_id)
            if stack_hash not in unique_stack_hashes:
                self._unique_stack_ids.append(stack_id)
                unique_stack_hashes.add(stack_hash)

    def add(self, stack_id: int, ts: int, label: bool):
        if stack_id in self.stack_events:
            raise ValueError("Stack already in this issue")
        self.stack_events[stack_id] = StackEvent(stack_id, ts, label)
        self._last_update_tss.append(ts)
        self._all_unique_stack_ids = None  # not actual anymore

    def remove(self, stack_id: int):
        self._last_update_tss.remove(self.stack_events[stack_id].ts)
        del self.stack_events[stack_id]
        self._all_unique_stack_ids = None  # not actual anymore

    def stack_ids(self, start_ts: int | None = None, end_ts: int | None = None, *, unique: bool = False) -> list[int]:
        if (start_ts is None) or (end_ts is None):
            # Try to get cached results in the common case.
            stack_ids = list(self.stack_events.keys())
            if unique:
                if self._all_unique_stack_ids is None:
                    # Save to cache.
                    self._all_unique_stack_ids = calculate_unique_stacks(stack_ids)
                stack_ids = self._all_unique_stack_ids
            return stack_ids

        # Arbitrary case, no cache is used.
        start_ts = start_ts or float("-inf")
        end_ts = end_ts or float("+inf")
        stack_ids = [event.stack_id for event in self.stack_events.values() if start_ts <= event.ts <= end_ts]
        if unique:
            stack_ids = calculate_unique_stacks(stack_ids)
        return stack_ids

    @property
    def size(self) -> int:
        return len(self.stack_events)

    @property
    def last_ts(self) -> int:
        return self._last_update_tss[-1]

    @staticmethod
    def from_events(id: int, events: dict[int, StackEvent]) -> "Issue":
        # TODO: refactoring
        ts = sorted([event.ts for event in events.values()])
        issue = Issue(id, ts[0])
        issue.stack_events = events
        issue._last_update_tss.extend(ts)
        return issue
