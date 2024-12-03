from pathlib import Path
import json
from datetime import datetime

import argparse
from ea.sim.dev.evaluation.objects import Prediction, ScoreRecord
from ea.sim.dev.scripts.training.losses import CircleLossUsingScores
from ea.sim.dev.scripts.training.models.on_tripets_with_similarity import ModelOnTripletsWithSimilarity
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
from loguru import logger

from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.loggers import ConsoleLogger
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.scripts.training.training.common import create_bucket_data, LabelFileNames, create_index_model
from ea.sim.dev.scripts.training.training.train_model import create_encoder, create_similarity
from ea.sim.main.configs import SeqCoderConfig
from ea.sim.main.data.buckets.bucket_data import DataSegment, BucketData
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.issue_scorer import MaxIssueScorer
from ea.sim.main.methods.neural.siam_network import SiamMultiModalModel
from ea.sim.main.methods.ranking_model import RankingModel
from ea.sim.main.methods.retrieval_model import CachedRetrievalModel, DummyRetrievalModel, IndexRetrievalModel
from ea.sim.main.methods.scoring_model import ScoringModel, SimpleScoringModel
from ea.sim.main.models_factory import create_seq_coder
from ea.sim.main.utils import Scope, device, ARTIFACTS_DIR
from ea.sim.dev.scripts.training.models.on_triplets import ModelOnTriplets
from ea.sim.dev.evaluation.evaluator import Evaluator
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.methods.neural.mix.lerch import TfIdfEncoder, WeightedIPSimilarity
from ea.sim.main.preprocess.entry_coders import Stack2Seq
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.tokenizers.simple import SimpleTokenizer

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    parser.add_argument("--model_ckpt_path", type=Path)

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

    model_name = 'lerch'
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

    train_stack_ids = data.all_reports(data_segment.train, unique_across_issues=not args.dup_attach)
    test_stack_ids = data.all_reports(data_segment.test, unique_across_issues=not args.dup_attach)
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

    all_items = (Item(tokens=coder(stack_id)) for stack_id in unsup_stacks)
    all_items = tqdm(all_items, total=len(unsup_stacks), desc="Fitting encoder")
    encoder = TfIdfEncoder(len(coder)).fit(all_items)
    similarity = WeightedIPSimilarity(len(coder))

    sim_stack_model = SiamMultiModalModel(coder, encoder.eval(), similarity.eval(), verbose=True).to(device)

    # Initializing index.
    segment_before_test = DataSegment(0, args.test_start)
    stacks_id_before_test = data.all_reports(segment_before_test, unique_across_issues=not args.dup_attach)
    index = create_index_model(sim_stack_model).fit(stacks_id_before_test)
    index.insert(test_stack_ids)
    logger.debug(f"Index is fitted on {len(train_stack_ids)} train stacks and {len(test_stack_ids)} test stacks")
    
    sim_stack_model.verbose = False  # not to disturb during testing

    rank_model = RankingModel(
        data=data,
        retrieval_model=IndexRetrievalModel(index, top_n=args.index_top_stacks),
        scoring_model=SimpleScoringModel(sim_stack_model),
        issue_scorer=MaxIssueScorer()
    )

    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", total=10_000)
    predictions_list  = [] 
    
    for prediction in rank_model.predict(events):
        prediction.stack_scores = sorted(prediction.stack_scores, key=lambda x: x.score, reverse=True)[:10]
        prediction.issue_scores = sorted(prediction.issue_scores, key=lambda x: x.score, reverse=True)[:10]
        predictions_list.append(prediction)

    predictions_json = json.dumps([prediction.to_dict() for prediction in predictions_list], indent=2)

    predictions_dir = ARTIFACTS_DIR / 'rank_model_predictions' / model_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (predictions_dir / "predictions.json").write_text(predictions_json)

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