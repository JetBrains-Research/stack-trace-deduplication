import argparse
from pathlib import Path

from ea.sim.dev.scripts.data.preprocess.so.preprocessor import Preprocessor
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import *

STEPS = [
    SelectSortedIssuesStep(),
    SelectIssuesWithMarkers(max_marked_reports_per_issue=1),
    MergeIssuesDuplicates(),
    # RemoveSimilarIssues()
]


def main(args: argparse.Namespace):
    sources = SlowOpsSources.load(args.data_folder)
    preprocessor = Preprocessor(sources, STEPS)
    preprocessor.run()
    preprocessor.save(args.output_folder)


if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--data_folder", type=Path, help="Folder with data for preprocessing.")
    _parser.add_argument("--output_folder", type=Path, help="Folder for saving preprocessed data.")
    _args = _parser.parse_args()
    main(_args)
