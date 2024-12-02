import argparse
import json

from ea.sim.dev.scripts.training.training.train_model import create_similarity
from loguru import logger
from tqdm import tqdm

from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.evaluation.evaluator import Evaluator
from ea.sim.dev.scripts.training.evaluating.openai.cache import CacheModel
from ea.sim.dev.scripts.training.training.common import create_bucket_data, LabelFileNames, create_index_model
from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.issue_scorer import MaxIssueScorer
from ea.sim.main.methods.ranking_model import RankingModel
from ea.sim.main.methods.retrieval_model import IndexRetrievalModel, CachedRetrievalModel
from ea.sim.main.methods.scoring_model import SimpleScoringModel, CachedScoringModel
from ea.sim.main.utils import ARTIFACTS_DIR, device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])

    parser.add_argument("--embeddings_folder", type=str, default=ARTIFACTS_DIR / "openai" / "v3-small")
    parser.add_argument("--model_name", type=str, default="openai_v3-small")

    return parser.parse_args()


def save_scores(args: argparse.Namespace) -> None:
    logger.debug("Predicting scores...")
    data_segment = Segment(
        train=DataSegment(args.train_start, args.train_longitude),
        val=DataSegment(args.val_start, args.val_longitude),
        test=DataSegment(args.test_start, args.test_longitude)
    )

    data = create_bucket_data(
        data_name=args.data_name,
        scope=args.scope,
        reports_dir=args.reports_dir,
        actions_path=True,
        state_path=args.labels_dir / LabelFileNames.state,
        forget_days=args.forget_days
    )

    Writer.initialize(args.artifacts_dir / "writer" / "train")
    HashStorage.initialize(data.stack_loader)

    train_stack_ids = data.all_reports(data_segment.train, unique_across_issues=not args.dup_attach)
    test_stack_ids = data.all_reports(data_segment.test, unique_across_issues=not args.dup_attach)

    # text-embedding-3-small: 512, 1536
    # text-embedding-3-large: 256, 1024, 3072
    similarity = create_similarity("cosine")
    sim_stack_model = CacheModel(args.embeddings_folder, similarity, trim=None).to(device)
    logger.debug(f"Model is loaded from {args.embeddings_folder}")

    # Initializing index.
    # segment_before_test = DataSegment(args.train_start, args.test_start - args.train_start)
    segment_before_test = DataSegment(0, args.test_start)
    stacks_id_before_test = data.all_reports(segment_before_test, unique_across_issues=not args.dup_attach)
    index = create_index_model(sim_stack_model).fit(stacks_id_before_test)
    index.insert(test_stack_ids)
    logger.debug(f"Index is fitted on {len(stacks_id_before_test)} train stacks and {len(test_stack_ids)} test stacks")
    sim_stack_model.verbose = False  # not to disturb during testing

    rank_model = RankingModel(
        data=data,
        retrieval_model=CachedRetrievalModel(IndexRetrievalModel(index, top_n=args.index_top_stacks)),
        scoring_model=CachedScoringModel(SimpleScoringModel(sim_stack_model)),
        issue_scorer=MaxIssueScorer()
    )

    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", dynamic_ncols=True, colour='green')
    predictions_list  = [] 
    
    for prediction in rank_model.predict(events):
        prediction.stack_scores = sorted(prediction.stack_scores, key=lambda x: x.score, reverse=True)[:10]
        prediction.issue_scores = sorted(prediction.issue_scores, key=lambda x: x.score, reverse=True)[:10]
        predictions_list.append(prediction)

    predictions_json = json.dumps([prediction.to_dict() for prediction in predictions_list], indent=2)

    predictions_dir = ARTIFACTS_DIR / 'rank_model_predictions' / args.model_name
    predictions_dir.mkdir(parents=True, exist_ok=True)
    (predictions_dir / "predictions.json").write_text(predictions_json)


def eval_scores(args: argparse.Namespace) -> None:
    logger.debug("Evaluating...")
    Evaluator(
        predictions_path=ARTIFACTS_DIR / 'rank_model_predictions' / args.model_name / "predictions.json",
        save_folder=ARTIFACTS_DIR / "eval" / args.model_name
    ).run()


def run():
    args = parse_args()
 
    config_path = ARTIFACTS_DIR / "config.json"
    config = json.loads(config_path.read_text())
    for arg, value in config.items():
        if hasattr(args, arg):
            value = args.__getattribute__(arg).__class__(value)
        args.__setattr__(arg, value)
 
    logger.debug(f"Setting up evaluator with args: {vars(args)}")
    save_scores(args)
    eval_scores(args)


if __name__ == "__main__":
    run()