import argparse
from ea.sim.main.methods.neural.encoders.objects import Item
from tqdm import tqdm
import json
from datetime import datetime

from loguru import logger

from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.scripts.training.training.common import create_bucket_data, LabelFileNames, create_index_model
from ea.sim.dev.scripts.training.training.train_model import create_encoder, create_similarity
from ea.sim.main.configs import SeqCoderConfig
from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.issue_scorer import MaxIssueScorer
from ea.sim.main.methods.neural.siam_network import SiamMultiModalModel
from ea.sim.main.methods.ranking_model import RankingModel
from ea.sim.main.methods.retrieval_model import CachedRetrievalModel, IndexRetrievalModel
from ea.sim.main.methods.scoring_model import SimpleScoringModel, CachedScoringModel
from ea.sim.main.models_factory import create_seq_coder
from ea.sim.main.utils import Scope, device, ARTIFACTS_DIR
 
from ea.sim.dev.scripts.training.models.on_pairs import ModelOnPairs
from ea.sim.dev.scripts.training.losses import InfoNCEPairs
from ea.sim.dev.evaluation.evaluator import Evaluator


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
    parser.add_argument("--model_ckpt_path", type=str, required=True)
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
    if args.random_init:
        model_name = "random_init"
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

    unsup_stacks = data.all_reports(data_segment.train, unique_across_issues=False)
    all_items = (Item(tokens=coder(stack_id)) for stack_id in unsup_stacks)
    token_encoder = "deep_crash" if "deep_crash" in str(args.model_ckpt_path) else "rnn"
    encoder = create_encoder("RNN", len(coder), token_encoder=token_encoder).fit(all_items, len(unsup_stacks))
    similarity = create_similarity("cosine")

    if not args.random_init:
        model = ModelOnPairs.load_from_checkpoint(args.model_ckpt_path, encoder=encoder, similarity=similarity, loss=InfoNCEPairs(), train_size=0, batch_size=1)
        logger.debug(f"Model is loaded from {args.model_ckpt_path}")
        # log time when file was last modified
        logger.debug(f"Model was last modified at {datetime.fromtimestamp(args.model_ckpt_path.stat().st_mtime)}")
    else:
        model = ModelOnPairs(encoder, similarity, InfoNCEPairs(), train_size=0, batch_size=1) # random init
        logger.debug(f"Model is randomly initialized")

    sim_stack_model = SiamMultiModalModel(coder, model.encoder.eval(), model.similarity.eval(), verbose=True).to(device)

    # Initializing index.
    segment_before_test = DataSegment(0, args.test_start)
    stacks_id_before_test = data.all_reports(segment_before_test, unique_across_issues=not args.dup_attach)
    index = create_index_model(sim_stack_model).fit(stacks_id_before_test)
    index.insert(test_stack_ids)
    logger.debug(f"Index is fitted on {len(train_stack_ids)} train stacks and {len(test_stack_ids)} test stacks")
    sim_stack_model.verbose = False  # not to disturb during testing

    scoring_model = CachedScoringModel(SimpleScoringModel(sim_stack_model))

    rank_model = RankingModel(
        data=data,
        retrieval_model=CachedRetrievalModel(IndexRetrievalModel(index, top_n=args.index_top_stacks)),
        scoring_model=scoring_model,
        issue_scorer=MaxIssueScorer()
    )

    events = data.get_events(data_segment.test, only_labeled=True, all_issues=False, with_dup_attach=args.dup_attach)
    events = tqdm(events, desc="Evaluating", total=10_000, dynamic_ncols=True, colour='green')
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
