import json

import argparse
from tqdm import tqdm
from loguru import logger

from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.scripts.training.training.common import create_bucket_data, LabelFileNames
from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.issue_scorer import MaxIssueScorer
from ea.sim.main.methods.ranking_model import RankingModel
from ea.sim.main.methods.retrieval_model import DummyRetrievalModel
from ea.sim.main.methods.scoring_model import SimpleScoringModel
from ea.sim.main.utils import Scope, ARTIFACTS_DIR
from ea.sim.dev.evaluation.evaluator import Evaluator
from ea.sim.main.preprocess.entry_coders import Stack2Seq
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.tokenizers.simple import SimpleTokenizer
from ea.sim.main.methods.classic.fast import FaST

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    return parser.parse_args()


def run():
    args = parse_args()
    config_path = ARTIFACTS_DIR / "config.json"
    config = json.loads(config_path.read_text())
    for arg, value in config.items():
        if hasattr(args, arg):
            value = args.__getattribute__(arg).__class__(value)
        args.__setattr__(arg, value)
    logger.debug(f"Setting up evaluator with args: {vars(args)}")

    model_name = 'fast'
    logger.info(f"Model name: {model_name}")

    if args.eval_only:
        Evaluator(
            predictions_path=ARTIFACTS_DIR / 'rank_model_predictions' / model_name / "predictions.json",
            save_folder=ARTIFACTS_DIR / "eval" / model_name
        ).run()
        return

    data_segment = Segment(
        train=DataSegment(args.train_start, args.train_longitude),
        val=DataSegment(args.val_start, args.val_longitude),
        test=DataSegment(args.test_start, args.test_longitude)
    )

    data = create_bucket_data(
        data_name=args.data_name,
        scope=args.scope,
        reports_dir=args.reports_dir,
        actions_path=True if args.scope != Scope.MainProject else args.labels_dir / LabelFileNames.actions,
        state_path=args.labels_dir / LabelFileNames.state,
        forget_days=args.forget_days
    )

    Writer.initialize(args.artifacts_dir / "writer" / "train")
    HashStorage.initialize(data.stack_loader)

    unsup_stacks = data.all_reports(data_segment.train, unique_across_issues=False)
    logger.debug(f"Preparing to fit neural model, {len(unsup_stacks)} unsup stacks")
    # coder = create_seq_coder(data.stack_loader, SeqCoderConfig.by_scope(scope)).partial_fit(unsup_stacks)

    coder = SeqCoder(
            data.stack_loader,
            Stack2Seq(cased=True, sep='.'),
            SimpleTokenizer(),
            max_len=None
    ).partial_fit(unsup_stacks)
    logger.debug("SeqCoder fitted or loaded")

    sim_stack_model = FaST(coder)
    sim_stack_model.fit(unsup_data=unsup_stacks)

    rank_model = RankingModel(
        data=data,
        retrieval_model=DummyRetrievalModel(),
        scoring_model=SimpleScoringModel(sim_stack_model),
        issue_scorer=MaxIssueScorer()
    )

    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", total=10_000, dynamic_ncols=True)
    predictions_list  = [] 
    
    for prediction in rank_model.predict(events):
        prediction.stack_scores = sorted(prediction.stack_scores, key=lambda x: x.score, reverse=True)[:10]
        prediction.issue_scores = sorted(prediction.issue_scores, key=lambda x: x.score, reverse=True)[:10]
        predictions_list.append(prediction)

    predictions_json = json.dumps([prediction.to_dict() for prediction in predictions_list], indent=2)

    predictions_dir = ARTIFACTS_DIR / 'rank_model_predictions' / model_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (predictions_dir / "predictions.json").write_text(predictions_json)

    rank_model.save_retrieval_cache(ARTIFACTS_DIR / "retrieval_cache" / model_name)

    Evaluator(
        predictions_path=ARTIFACTS_DIR / 'rank_model_predictions' / model_name / "predictions.json",
        save_folder=ARTIFACTS_DIR / "eval" / model_name
    ).run()

    reranking_time, scoring_time = rank_model.get_average_times()
    time_info = {
        "reranking_time": reranking_time,
        "scoring_time": scoring_time
    }
    file_to_save = ARTIFACTS_DIR / 'time_info' / model_name / "time_info.json"
    file_to_save.parent.mkdir(parents=True, exist_ok=True)
    with open(file_to_save, 'w') as f:
        json.dump(time_info, f)


if __name__ == "__main__":
    run()
