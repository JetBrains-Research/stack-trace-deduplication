import argparse
import json
from pathlib import Path
from typing import NamedTuple, Iterable, Any

import pandas as pd
from tqdm import tqdm

from ea.sim.main.data.objects.stack import Stack, Frame

TRACES_FILE_NAME = "netbeans_stacktraces.json"
REPORTS_FOLDER = "reports"
STATE_FILE_NAME = "state.csv"


class AttachEvent(NamedTuple):
    timestamp: int
    report_id: int
    issue_id: int

    columns = ["timestamp", "rid", "iid"]


def save_stack(stack: Stack, folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / f"{stack.id}.json"
    file_path.write_text(json.dumps(stack.to_dict(), indent=2))


def save_state(events: list[AttachEvent], folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=events, columns=AttachEvent.columns) \
        .sort_values(by="timestamp") \
        .to_csv(folder / STATE_FILE_NAME, index=False)


def parse_stack(report_jdict: dict[str, Any]) -> Stack:
    stacktrace = report_jdict["stacktrace"][0]
    frames = [
        Frame(
            name=frame_jdict["function"],
            file_name=frame_jdict["file"],
            line_number=frame_jdict["fileline"],
        )
        for frame_jdict in stacktrace["frames"]
    ]

    return Stack(
        id=report_jdict["bug_id"],
        timestamp=int(report_jdict["creation_ts"] * 1_000),
        errors=report_jdict["exception"],
        frames=frames,
        messages=None,
        comment=report_jdict["comments"]
    )


def parse(file_path: Path) -> Iterable[tuple[Stack, AttachEvent]]:
    traces_jdict = json.loads(file_path.read_text())
    for report_jdict in tqdm(traces_jdict):
        stack = parse_stack(report_jdict)
        event = AttachEvent(
            timestamp=int(report_jdict["creation_ts"] * 1_000),
            report_id=report_jdict["bug_id"],
            issue_id=report_jdict["dup_id"] or report_jdict["bug_id"]
        )
        yield stack, event


def main(args: argparse.Namespace):
    file_path = args.data_folder / TRACES_FILE_NAME
    reports_folder = args.output_folder / REPORTS_FOLDER

    events = []
    for stack, event in parse(file_path):
        save_stack(stack, reports_folder)
        events.append(event)

    save_state(events, args.output_folder)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--data_folder", type=Path, help="Folder with data for preprocessing.")
    _parser.add_argument("--output_folder", type=Path, help="Folder for saving preprocessed data.")
    _args = _parser.parse_args()
    main(_args)
