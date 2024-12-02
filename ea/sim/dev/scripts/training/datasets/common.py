from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from enum import Enum


@dataclass
class State:
    report_ids: list[int]
    group_ids: list[int]

    def __len__(self) -> int:
        return len(self.report_ids)


def load_state(file_path: Path) -> State:
    df = pd.read_csv(file_path)
    return State(
        report_ids=df.rid.tolist(),
        group_ids=df.iid.tolist(),
    )


def load_groups(state: State) -> dict[int, list[int]]:
    groups = {}
    for report_id, group_id in zip(state.report_ids, state.group_ids):
        if group_id not in groups:
            groups[group_id] = []
        groups[group_id].append(report_id)
    return groups


class SamplingTechnique(Enum):
    ALL_WITH_ALL = "all_with_all"
    LINEAR = "linear" # Form groups (report_id_1, report_id_2), (report_id_3, report_id_4)...
