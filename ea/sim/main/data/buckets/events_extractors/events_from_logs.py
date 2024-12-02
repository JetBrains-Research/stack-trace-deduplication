"""
Events from "attach / detach" actions.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ea.sim.main.data.buckets.event_state_model import StackAdditionEvent
from ea.sim.main.data.buckets.events_extractors import EventsData


class EventsFromLogs(EventsData):
    def __init__(self, actions_path: Path, sec_denom: int):
        self.actions_path = actions_path  # .csv file (timestamp, rid, iid, label)
        self.sec_denom = sec_denom

    def _actions_data(self) -> np.ndarray:
        actions_df = pd.read_csv(self.actions_path)
        return actions_df.values

    def events(self) -> list[StackAdditionEvent]:
        actions = self._actions_data()
        first_ts = datetime.fromtimestamp(actions[0, 0] // self.sec_denom)
        return [
            StackAdditionEvent(
                id=event_id,
                stack_id=stack_id,
                issue_id=issues_id,
                ts=(datetime.fromtimestamp(ts // self.sec_denom) - first_ts).total_seconds() / self.day_secs,
                label=label
            )
            for event_id, (ts, stack_id, issues_id, label) in enumerate(tqdm(actions))
        ]
