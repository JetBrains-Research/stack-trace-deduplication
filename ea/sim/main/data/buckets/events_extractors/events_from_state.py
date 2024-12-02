"""
Events from "report -> issue" system state.
"""
from abc import abstractmethod, ABC
from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from ea.sim.main.data.buckets.event_state_model import StackAdditionEvent
from ea.sim.main.data.buckets.events_extractors import EventsData
from ea.sim.main.utils import StackId, IssueId, Scope


class LabelExtractor(ABC):
    @abstractmethod
    def label(self, stack_id: StackId, issue_id: IssueId) -> bool:
        raise NotImplementedError

    @staticmethod
    def from_value(value: bool | Path) -> "LabelExtractor":
        if isinstance(value, bool):
            return DefaultLabelExtractor(value)
        elif isinstance(value, Path):
            return ActionsLabelExtractor(value)
        else:
            raise ValueError(f"Incorrect type for value: '{isinstance(value)}'")


class DefaultLabelExtractor(LabelExtractor):
    def __init__(self, label: bool):
        self._label = label

    def label(self, stack_id: StackId, issue_id: IssueId) -> bool:
        return self._label


class ActionsLabelExtractor(LabelExtractor):
    def __init__(self, actions_path: Path):
        actions_df = pd.read_csv(actions_path.expanduser().resolve())
        self._labeled_events = {
            (rid, iid)
            for (rid, iid, label) in actions_df[["rid", "iid", "label"]].itertuples(index=False)
            if label
        }

    def label(self, stack_id: StackId, issue_id: IssueId) -> bool:
        return (stack_id, issue_id) in self._labeled_events


class EventsFromState(EventsData):
    def __init__(
            self,
            state_path: Path,
            labels_source: bool | Path,
            existed_reports: set[int],
            sec_denom: int,
            scope: Scope
    ):
        self.state_path = state_path.expanduser()  # .csv file (ts, rid, iid)
        self.label_extractor = LabelExtractor.from_value(labels_source)

        self.sec_denom = sec_denom
        self.scope = scope
        self.existed_reports = existed_reports

    def days_diff(self, from_timestamp: int, to_timestamp: int) -> int:
        return int((to_timestamp - from_timestamp) / self.sec_denom / 60 / 60 / 24)

    def load_state(self) -> pd.DataFrame:
        state_df = pd.read_csv(self.state_path).sort_values(by="timestamp")
        return state_df[["timestamp", "rid", "iid"]]  # .values

    def events(self) -> list[StackAdditionEvent]:
        state = self.load_state()
        start_ts = self.start_timestamp(self.scope)
        if start_ts is not None:
            state = state[state.timestamp >= start_ts]
            logger.debug(
                f"Selecting events from {datetime.fromtimestamp(start_ts // self.sec_denom)} date "
                f"for '{self.scope.value}' scope"
            )
        else:
            logger.debug(f"Selecting all events (no start date provided) for '{self.scope.value}' scope")

        first_ts = state.timestamp.min()
        state = state[state.rid.isin(self.existed_reports)]
        return [
            StackAdditionEvent(
                id=event_id,
                stack_id=rid,
                issue_id=iid,
                ts=self.days_diff(first_ts, ts),
                label=self.label_extractor.label(rid, iid)
            )
            for (event_id, ts, rid, iid) in state.itertuples(index=True)
        ]
