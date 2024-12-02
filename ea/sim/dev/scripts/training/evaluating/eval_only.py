import argparse
from pathlib import Path
import json

from loguru import logger

from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.main.utils import ARTIFACTS_DIR
from ea.sim.dev.evaluation.evaluator import Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    parser.add_argument("--model_name", type=Path, default="embedding_model_rnn_infonce_drop05_max100") 
    # parser.add_argument("--model_name", type=Path, default="cross_enc_v6_1_100_200_100_0.2_rnn_bce_with_logits") 
    # parser.add_argument("--model_name", type=Path, default="openai_v3-small") 
    # parser.add_argument("--model_name", type=Path, default="deep_crash_drop05") 
    # parser.add_argument("--model_name", type=Path, default="fast") 
    # parser.add_argument("--model_name", type=Path, default="s3m") 
    # parser.add_argument("--model_name", type=Path, default="lerch") 

    return parser.parse_args()


def run():
    args = parse_args()
    logger.info(f"Model name: {args.model_name}")

    Evaluator(
        predictions_path=ARTIFACTS_DIR / 'rank_model_predictions' / args.model_name / "predictions.json",
        save_folder=ARTIFACTS_DIR / "eval" / args.model_name
    ).run()


if __name__ == "__main__":
    run()
