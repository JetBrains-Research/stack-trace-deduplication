from argparse import ArgumentParser
from pathlib import Path

from ea.sim.main.utils import Scope


def setup_model_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int, required=False, default=42)

    parser.add_argument('--hyp_top_issues', type=int, default=None)
    parser.add_argument('--hyp_top_stacks', type=int, default=None)
    parser.add_argument('--index_top_stacks', type=int, default=50)
    return parser


def setup_train_markup_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_longitude', type=int, default=0)

    parser.add_argument('--val_start', type=int, default=0)
    parser.add_argument('--val_longitude', type=int, default=0)

    parser.add_argument('--test_start', type=int, default=0)
    parser.add_argument('--test_longitude', type=int, default=0)
    parser.add_argument('--max_per_group', type=int, default=100)

    parser.add_argument('--data_name', type=str, help='Data name', default='netbeans')
    parser.add_argument('--forget_days', type=int, default=365)
    parser.add_argument('--random_init', type=bool, default=False)
    parser.add_argument('--eval_only', type=bool, default=False)

    return parser


def setup_data_parser() -> ArgumentParser:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--labels_dir", type=Path,
        help="Directory with actions, state, constraints and other files connected to labels",
        default=Path("")
    )
    parser.add_argument(
        "--reports_dir", type=Path, help="Directory with json reports",
        default=Path("")
    )
    parser.add_argument(
        "--dataset_dir", type=Path, help="Directory with triplets for training",
        default=Path("")
    )
    parser.add_argument(
        "--scope", type=Scope, choices=list(Scope), help="Scope of reports", default=Scope.NetBeans
    )
    parser.add_argument(
        "--artifacts_dir", type=Path,
        default=Path(""),
    )
    return parser
