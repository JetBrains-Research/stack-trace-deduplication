import argparse
from pathlib import Path
from ea.sim.dev.scripts.training.models.cross_encoder_on_triplets import CrossEncoderModel
from ea.sim.main.data.buckets.event_state_model import StackAdditionState
from ea.sim.main.methods.neural.cross_encoders.base import SimStackModelFromCrossEncoder
from ea.sim.main.methods.neural.cross_encoders.s3m import S3M
from ea.sim.main.methods.neural.encoders.objects import Item
from ea.sim.main.preprocess.entry_coders import Stack2Seq
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.tokenizers.simple import SimpleTokenizer
from tqdm import tqdm
import json
from datetime import datetime
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
from ea.sim.main.methods.scoring_model import ScoringModel, SimpleScoringModel, CachedScoringModel
from ea.sim.main.models_factory import create_seq_coder
from ea.sim.main.utils import Scope, device, ARTIFACTS_DIR
 
from ea.sim.dev.scripts.training.models.on_pairs import ModelOnPairs
from ea.sim.dev.scripts.training.datasets.pair import PairDataset
from ea.sim.dev.scripts.training.losses import InfoNCEPairs, RankNetLoss
from ea.sim.dev.evaluation.evaluator import Evaluator
from ea.sim.dev.scripts.training.models.on_triplets import ModelOnTriplets
from ea.sim.main.methods.S3M_mock_retrival import S3MMockRetrievalModel


class StreamingJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)

def stream_json(data, file_path):
    with open(file_path, 'w') as f:
        f.write('[')
        first = True
        for item in data:
            if not first:
                f.write(', ')
            else:
                first = False
            json.dump(item, f, cls=StreamingJSONEncoder)
        f.write(']')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    parser.add_argument("--model_ckpt_path", type=Path, required=True)

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

    model_name = args.model_ckpt_path.parent.name
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

    coder = SeqCoder(
            data.stack_loader,
            Stack2Seq(cased=True, sep='.'),
            SimpleTokenizer(),
            max_len=None
    ).partial_fit(unsup_stacks)
    logger.debug("SeqCoder fitted or loaded")

    cross_encoder = S3M(evaluating=True, vocab_size=len(coder), 
                        token_emb_dim=100, hidden_size=200
                        ).to(device)

    cross_encoder_on_triplets = CrossEncoderModel.load_from_checkpoint(args.model_ckpt_path, encoder=cross_encoder, scope=args.scope,
                                                                       loss=None, train_size=0, batch_size=0, epochs=0).to(device)
    sim_stack_cross_model = SimStackModelFromCrossEncoder(cross_encoder_on_triplets.encoder.eval(), coder)
    
    rank_model = RankingModel(
        data=data,
        retrieval_model=DummyRetrievalModel(),
        scoring_model=SimpleScoringModel(sim_stack_cross_model),
        issue_scorer=MaxIssueScorer(),
    )

    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", total=10_000, dynamic_ncols=True, colour='green')
    predictions_list  = [] 
    
    for prediction in rank_model.predict(events):
        prediction.stack_scores = sorted(prediction.stack_scores, key=lambda x: x.score, reverse=True)[:10]
        prediction.issue_scores = sorted(prediction.issue_scores, key=lambda x: x.score, reverse=True)

        if not prediction.is_new_issue:
            num_equal = 1
            while num_equal < len(prediction.issue_scores) and prediction.issue_scores[num_equal].score == prediction.issue_scores[0].score:
                num_equal += 1
            ids_equal = [prediction.issue_scores[i].object_id for i in range(num_equal)]
            if prediction.target_id in ids_equal:
                target_index = ids_equal.index(prediction.target_id)
                prediction.issue_scores[0], prediction.issue_scores[target_index] = prediction.issue_scores[target_index], prediction.issue_scores[0]
            
        prediction.issue_scores = prediction.issue_scores[:10]

        predictions_list.append(prediction)

    predictions_json = json.dumps([prediction.to_dict() for prediction in predictions_list], indent=2)

    predictions_dir = ARTIFACTS_DIR / 'rank_model_predictions' / model_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    (predictions_dir / "predictions.json").write_text(predictions_json)

    logger.info(f"Model name: {model_name}")

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

    # path_to_predictions = ARTIFACTS_DIR / 'predictions' / model_name / "predictions.json"
    # predictions = rank_model.predictions
    # path_to_predictions.parent.mkdir(parents=True, exist_ok=True)
    # with open(path_to_predictions, 'w') as f:
    #     json.dump(predictions, f)
    # logger.info(f"Predictions are saved to {path_to_predictions}")


if __name__ == "__main__":
    run()
