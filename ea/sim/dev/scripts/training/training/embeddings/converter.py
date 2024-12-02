import json
from pathlib import Path
from typing import NamedTuple


class Report(NamedTuple):
    id: int
    text: str


class Converter:
    EMPTY: str = " "

    def __init__(self, folder: Path):
        self.paths = Converter.get_report_paths(folder)
        print(f"Total {len(self.paths)} reports found in '{folder}'.")

    def get_text(self, report_id: int) -> str:
        if report_id not in self.paths:
            return Converter.EMPTY

        file_path = self.paths[report_id]
        jdict = json.loads(file_path.read_text())
        frames = [frame for frame in jdict["elements"][::-1]]
        names = [frame["name"] for frame in frames]
        text = " -> ".join(names).strip()
        if not text:
            text = Converter.EMPTY
        return text

    @staticmethod
    def get_report_id(path: Path) -> int:
        file_name = path.name
        report_id, _ = file_name.split(".")
        return int(report_id)

    @staticmethod
    def get_report_paths(folder: Path) -> dict[int, Path]:
        return {
            Converter.get_report_id(file_path): file_path
            for file_path in folder.rglob("*.json")
        }
