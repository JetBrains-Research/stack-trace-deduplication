import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

from ea.sim.main.data.buckets.issues_selector import ReportTimeIssueSelector
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.data.objects.issue import Issue
from ea.sim.main.utils import IssueId, StackId


class StackAdditionEvent(NamedTuple):
    id: int
    stack_id: int
    issue_id: int
    ts: int
    label: bool


@dataclass(frozen=True)
class StackAdditionState:
    id: int
    stack_id: int
    issues: dict[int, Issue]
    issue_id: int
    label: bool

    @property
    def is_new_issue(self) -> bool:
        return self.issue_id not in self.issues

    @staticmethod
    def from_event(event: StackAdditionEvent, issues: dict[int, Issue]) -> 'StackAdditionState':
        return StackAdditionState(event.id, event.stack_id, issues, event.issue_id, event.label)


class StateModel:
    def __init__(self):
        self._last_ts: int | None = None
        self.issues: dict[IssueId, Issue] = dict()
        self.stacks: dict[StackId, IssueId] = dict()
        self.timestamps: dict[StackId, int] = dict()
        self._actual_issues: set[IssueId] = set()

    def _detach(self, event: StackAdditionEvent):
        issue_id = self.stacks[event.stack_id]
        self.issues[issue_id].remove(event.stack_id)
        del self.stacks[event.stack_id]

    def _attach(self, event: StackAdditionEvent):
        if event.issue_id not in self.issues:
            self.issues[event.issue_id] = Issue(event.issue_id, event.ts)
        self.issues[event.issue_id].add(event.stack_id, event.ts, event.label)
        self.stacks[event.stack_id] = event.issue_id
        self.timestamps[event.stack_id] = event.ts
        self._actual_issues.add(event.issue_id)

    def update(self, event: StackAdditionEvent):
        self._last_ts = event.ts
        if event.stack_id in self.stacks:
            self._detach(event)
        self._attach(event)


class EventStateModel:
    def __init__(self, name: str, forget_days: float | None = None, verbose: bool = False):
        self.name = name
        self.state = StateModel()
        self.issues_selector = ReportTimeIssueSelector(forget_days)
        self.verbose = verbose

    def warmup(self, actions: Iterable[StackAdditionEvent]):
        for action in actions:
            self.state.update(action)

    def select_issues(self, current_day: int, all_issues: bool = False) -> dict[int, Issue]:
        if all_issues:
            return self.state.issues
        return self.issues_selector.select(self.state.issues, current_day)

    def _does_event_satisfy(
            self,
            event: StackAdditionEvent,
            *,
            only_labeled: bool,
            with_dup_attach: bool
    ) -> bool:
        def is_attach_action() -> bool:
            return (event.issue_id != -1)# and (event.issue_id != event.stack_id)

        def does_label_satisfy() -> bool:
            return (not only_labeled) or event.label

        def does_dup_satisfy() -> bool:
            satisfy = True
            if not with_dup_attach:
                hash_storage = HashStorage.get_instance()
                all_stack_ids = self.all_seen_stacks(only_unique_from_issue=False)
                hashes = set(hash_storage.hashes(all_stack_ids))
                satisfy = hash_storage.hash(event.stack_id) not in hashes
            return satisfy

        # Lazy "AND" expression.
        return is_attach_action() and does_label_satisfy() and does_dup_satisfy()

    def collect(
            self,
            actions: Iterable[StackAdditionEvent],
            *,
            only_labeled: bool,
            all_issues: bool,
            with_dup_attach: bool
    ) -> Iterable[StackAdditionState]:
        for action in actions:
            assert action.issue_id > 0, "Can't handle detach events right now"
            if self._does_event_satisfy(action, only_labeled=only_labeled, with_dup_attach=with_dup_attach):
                current_issues = self.select_issues(action.ts, all_issues=all_issues)
                event = StackAdditionState.from_event(action, current_issues)
                yield event

            self.state.update(action)

    def all_seen_stacks(
            self,
            start_ts: int | None = None,
            end_ts: int | None = None,
            *,
            only_unique_from_issue: bool = False
    ) -> list[StackId]:
        all_issues = self.state.issues.values()
        return [
            stack_id
            for issue in all_issues
            for stack_id in issue.stack_ids(start_ts, end_ts, unique=only_unique_from_issue)
        ]

    def file_path(self, days_num: float) -> Path:
        # days_num = days_num or self.warmup_days
        forget_days = self.issues_selector.forget_days or 0
        event_states_dir = Path(__file__).parent / "event_states"
        file_path = event_states_dir / f"{self.name}_event_state_days_{days_num}_forget_{forget_days}.pkl"
        return file_path

    def load(self, days_num: int) -> "EventStateModel":
        tmp_dict = pickle.loads(self.file_path(days_num).read_bytes())
        self.state = tmp_dict["state"]
        return self

    def save(self, days_num: float):
        file_path = self.file_path(days_num)
        file_path.parent.mkdir(exist_ok=True, parents=True)
        file_path.write_bytes(pickle.dumps(self.__dict__))
