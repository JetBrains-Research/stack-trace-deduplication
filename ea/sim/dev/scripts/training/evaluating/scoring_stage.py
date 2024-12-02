import argparse
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pandas as pd
 
from loguru import logger
 
from ea.sim.dev.evaluation.evaluator import Evaluator
from ea.sim.dev.scripts.training.models.cross_encoder_on_triplets import CrossEncoderModel
from ea.sim.main.methods.neural.cross_encoders.base import SimStackModelFromCrossEncoder
from ea.sim.main.methods.neural.cross_encoders.rnn import LSTMCrossEncoderConfig, \
    LSTMCrossEncoder
from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.loggers import ConsoleLogger
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.scripts.training.training.common import create_bucket_data, LabelFileNames
from ea.sim.dev.scripts.training.training.train_model import create_encoder, create_similarity
from ea.sim.main.configs import SeqCoderConfig
from ea.sim.main.data.buckets.bucket_data import DataSegment, BucketData
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.issue_scorer import MaxIssueScorer
from ea.sim.main.methods.neural.siam_network import SiamMultiModalModel
from ea.sim.main.methods.ranking_model import RankingModel
from ea.sim.main.methods.retrieval_model import CachedRetrievalModel, IndexRetrievalModel
from ea.sim.main.methods.scoring_model import ScoringModel, SimpleScoringModel
from ea.sim.main.models_factory import create_seq_coder
from ea.sim.main.utils import Scope, device, ARTIFACTS_DIR
from ea.sim.dev.scripts.training.reranker_losses import RerankerLoss, BCELossWithLogits, PairwiseSoftmaxCrossEntropyLoss
from torch.utils.data import DataLoader
from ea.sim.dev.scripts.training.datasets.triplet import TripletDataset
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    parser.add_argument("--cross_encoder_path", type=Path, required=True)
    return parser.parse_args()

 
def save_score_distridution(args, coder, cross_encoder_on_triplets, cross_encoder_path):
    val_dataset = TripletDataset(
        file_path=args.dataset_dir / "val.csv",
        seq_coder=coder,
        max_per_group=5,
        random_size=1000
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=val_dataset.collate_fn)
    all_pos_scores = []
    all_neg_scores = []
    cross_encoder_on_triplets = cross_encoder_on_triplets.to(device)
    for batch in val_loader:
        positive_scores = cross_encoder_on_triplets.compute_positive_scores(batch)
        negative_scores = cross_encoder_on_triplets.compute_negative_scores(batch)
        all_pos_scores.extend(positive_scores.cpu().detach().numpy().tolist())
        all_neg_scores.extend(negative_scores.cpu().detach().numpy().tolist())
    path_to_save = ARTIFACTS_DIR / 'tmp' / 'scores' / Path(cross_encoder_path).parent.name
    path_to_save.mkdir(parents=True, exist_ok=True)
    with open(path_to_save / 'positive_scores.json', 'w') as f:
        json.dump(all_pos_scores, f)
    with open(path_to_save / 'negative_scores.json', 'w') as f:
        json.dump(all_neg_scores, f)


def run():
    args = parse_args()
    config_path = ARTIFACTS_DIR / "config.json"
    config = json.loads(config_path.read_text())
    for arg, value in config.items():
        if hasattr(args, arg):
            value = args.__getattribute__(arg).__class__(value)
        args.__setattr__(arg, value)
    logger.debug(f"Setting up evaluator with args: {vars(args)}")
 
    cross_encoder_path = args.cross_encoder_path

    model_name = cross_encoder_path.parent.name
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
    coder = create_seq_coder(data.stack_loader, SeqCoderConfig.by_scope(Scope.MainProject)).partial_fit(train_stack_ids)
 
    cross_encoder_config = LSTMCrossEncoderConfig(token_encoder="rnn", dropout=0.1, d_input=100, hidden_size=200, output_size=100, loss=PairwiseSoftmaxCrossEntropyLoss())
 
    cross_encoder = LSTMCrossEncoder(cross_encoder_config).to(device)

    cross_encoder_on_triplets = CrossEncoderModel.load_from_checkpoint(cross_encoder_path, encoder=cross_encoder, scope=args.scope,
                                                                       loss=cross_encoder_config.loss, train_size=0, batch_size=0, epochs=0).to(device)
    logger.debug(f"Cross encoder is loaded from {cross_encoder_path}")
    logger.debug(f"Model was last modified at {datetime.fromtimestamp(cross_encoder_path.stat().st_mtime)}")
 
 
    sim_stack_cross_model = SimStackModelFromCrossEncoder(cross_encoder_on_triplets.encoder.eval(), coder)
 
    rank_model = RankingModel(
        data=data,
        retrieval_model=CachedRetrievalModel(IndexRetrievalModel(None, top_n=args.index_top_stacks)),
        scoring_model=SimpleScoringModel(sim_stack_cross_model),
        issue_scorer=MaxIssueScorer(),
    )
    emb_model_name = "embedding_model_rnn_infonce_drop05_max100"
    rank_model.load_retrieval_cache(ARTIFACTS_DIR / "retrieval_cache" / emb_model_name)
    logger.info(f"Retrieval cache is loaded from {ARTIFACTS_DIR / 'retrieval_cache' / emb_model_name}")
 
    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", total=10_000, dynamic_ncols=True, colour="green")
    
    predictions_list  = [] 
    
    for prediction in rank_model.predict(events):
        prediction.stack_scores = sorted(prediction.stack_scores, key=lambda x: x.score, reverse=True)[:10]
        prediction.issue_scores = sorted(prediction.issue_scores, key=lambda x: x.score, reverse=True)[:10]
        predictions_list.append(prediction)

    predictions_json = json.dumps([prediction.to_dict() for prediction in predictions_list], indent=2)
    predictions_dir = ARTIFACTS_DIR / 'rank_model_predictions' / model_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (predictions_dir / "predictions.json").write_text(predictions_json)

    logger.info(f"Model name: {model_name}")
    logger.info(f"Predictions are saved to {predictions_dir / 'predictions.json'}")

    rank_model.save_retrieval_cache(ARTIFACTS_DIR / "retrieval_cache" / model_name)

    Evaluator(
        predictions_path=ARTIFACTS_DIR / 'rank_model_predictions' / model_name / "predictions.json",
        save_folder=ARTIFACTS_DIR / "eval" / model_name
    ).run()

    reranking_time, scoring_time = rank_model.get_average_times()
    time_info = {
        "reranking_time": reranking_time,
        "scoring_time": scoring_time,
        "sum": (reranking_time + scoring_time) * 1000
    }
    file_to_save = ARTIFACTS_DIR / 'time_info' / model_name / "time_info.json"
    file_to_save.parent.mkdir(parents=True, exist_ok=True)
    with open(file_to_save, 'w') as f:
        json.dump(time_info, f)
 
 
if __name__ == "__main__":
    run()
