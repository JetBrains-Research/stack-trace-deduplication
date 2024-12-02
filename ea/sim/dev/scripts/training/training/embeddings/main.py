import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ea.sim.dev.scripts.training.training.embeddings.client import OpenAIClient, PriceTracker, Request
from ea.sim.dev.scripts.training.training.embeddings.converter import Report, Converter
from ea.sim.dev.scripts.training.training.embeddings.models import OpenAIModel

API_KEY = os.environ.get("OPENAI_API_KEY")

assert API_KEY is not None


def get_texts(folder: Path, path: Path) -> Iterable[Report]:
    report_ids = pd.read_csv(path)["rid"]
    converter = Converter(folder)
    for report_id in sorted(report_ids):
        yield Report(report_id, converter.get_text(report_id))


def main(reports_folder: Path, reports_path: Path, save_folder: Path, model_name: str):
    model = OpenAIModel.by_name(model_name)
    client = OpenAIClient(model.value, API_KEY)
    price_tracker = PriceTracker(model, alert_every_spent_dollar=1)
    embeddings, hashes = dict(), dict()

    report_ids = pd.read_csv(reports_path)["rid"]
    reports_cnt = len(report_ids)
    for report_id, report_text in tqdm(get_texts(reports_folder, reports_path), total=reports_cnt):
        text_hash = hash(report_text)
        if text_hash not in hashes:
            hashes[text_hash] = report_id
            request = Request(report_text)
            response = client.encode(request)
            embeddings[report_id] = response.embedding
            price_tracker.update(response.consumed_tokens)
        else:
            dup_report_id = hashes[text_hash]
            embeddings[report_id] = embeddings[dup_report_id]

    report_ids = sorted(embeddings.keys())
    embeddings = np.vstack([embeddings[report_id] for report_id in report_ids])

    save_folder = save_folder / model_name
    save_folder.mkdir(parents=True, exist_ok=True)
    (save_folder / "hashes.json").write_text(json.dumps(hashes, indent=2))
    (save_folder / "report_ids.json").write_text(json.dumps(report_ids, indent=2))
    np.savez(save_folder / "embeddings.npz", embeddings=embeddings)

    print(f"Embeddings are saved to {save_folder}")
    price_tracker.print_price()


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--reports_folder", type=Path)
    _parser.add_argument("--reports_path", type=Path)
    _parser.add_argument("--save_folder", type=Path)
    _parser.add_argument("--model_name", type=str, default="v3-small")
    _args = _parser.parse_args()
    print(f"Parsed args: {_args}")
    main(_args.reports_folder, _args.reports_path, _args.save_folder, _args.model_name)
