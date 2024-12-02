from pathlib import Path
from typing import Set
from tqdm import tqdm

import pandas as pd
import argparse


def save_missed_reports(report_ids: Set[int], save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    report_ids = sorted(report_ids)
    with (save_dir / "missed_reports.txt").open("w") as file:
        for report_id in report_ids:
            file.write(f"{report_id}\n")


def get_existed_reports(reports_dir: Path) -> Set[int]:
    report_ids = []

    for file_path in tqdm(reports_dir.glob("*.json"), desc="Getting existed reports"):
        file_name = file_path.name
        report_id, _ = file_name.split(".")
        report_ids.append(int(report_id))

    return set(report_ids)


def get_required_reports(reports_path: Path) -> Set[int]:
    df = pd.read_csv(reports_path)
    return set(df.rid)


def main(reports_path: Path, reports_dir: Path, save_dir: Path):
    existed_reports = get_existed_reports(reports_dir)
    required_reports = get_required_reports(reports_path)
    missed_reports = required_reports - existed_reports

    print("Total existed reports:", len(existed_reports))
    print("Total required reports:", len(required_reports))
    print("Total missed reports:", len(missed_reports))

    save_missed_reports(missed_reports, save_dir)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--reports_path", type=Path)
    _parser.add_argument("--reports_dir", type=Path)
    _parser.add_argument("--save_dir", type=Path)
    _args = _parser.parse_args()
    main(_args.reports_path, _args.reports_dir, _args.save_dir)
