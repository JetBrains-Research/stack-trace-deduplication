import argparse
import typing as tp
from pathlib import Path

from ea.sim.main.methods.neural.cross_encoders.s3m import S3M
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from loguru import logger
from torch.utils.data import DataLoader
import json

from ea.sim.main.preprocess.entry_coders import Stack2Seq
from ea.sim.main.preprocess.seq_coder import SeqCoder
from ea.sim.main.preprocess.tokenizers.simple import SimpleTokenizer
from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.common.utils import set_seed, random_seed
from ea.sim.dev.scripts.training.common.arg_parsers import setup_model_parser, setup_train_markup_parser, \
    setup_data_parser
from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.dev.scripts.training.callbacks import CallbackArgs, callback_factory
from ea.sim.dev.scripts.training.datasets.triplet import TripletDataset
from ea.sim.dev.scripts.training.training.common import create_bucket_data, fit_hash_storage, \
    LabelFileNames
from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.methods.neural.similarity import CosineSimilarity, Similarity
 
from ea.sim.dev.scripts.training.models.cross_encoder_on_triplets import CrossEncoderModel
from ea.sim.main.utils import Scope, device, ARTIFACTS_DIR
 
from ea.sim.dev.scripts.training.datasets.common import SamplingTechnique
from ea.sim.dev.scripts.training.reranker_losses import RankNetRerankerLoss


class PathArgs(tp.NamedTuple):
    actions_path: Path
    state_path: Path
    reports_dir: Path
    dataset_dir: Path
    artifacts_dir: Path | None


def create_similarity(sim_type: str) -> Similarity:
    if sim_type == "cosine":
        return CosineSimilarity()
    else:
        raise ValueError(f"Unknown similarity type: {sim_type}")


def train_similarity_model(
        data_name: str, scope: Scope,
        data_segment: Segment, path_args: PathArgs, callback_args: CallbackArgs,
        forget_days: int | None, max_per_group: int, path_to_save: Path
):
    data = create_bucket_data(
        data_name, scope, path_args.reports_dir, True if scope != Scope.MainProject else path_args.actions_path,
        path_args.state_path,
        forget_days=forget_days
    )
    Writer.initialize(path_args.artifacts_dir / "writer" / "train")
    HashStorage.initialize(data.stack_loader)
    fit_hash_storage(data, data_segment)  # will be skipped if already fitted

    unsup_stacks = data.all_reports(data_segment.train, unique_across_issues=False)
    logger.debug(f"Preparing to fit neural model, {len(unsup_stacks)} unsup stacks")
    coder = SeqCoder(
            data.stack_loader,
            Stack2Seq(cased=True, sep='.'),
            SimpleTokenizer(),
            max_len=None
    ).partial_fit(unsup_stacks)
    logger.debug("SeqCoder fitted or loaded")

    train_dataset = TripletDataset(
        file_path=path_args.dataset_dir / "train.csv",
        seq_coder=coder,
        max_per_group=max_per_group,
        sampling_technique=SamplingTechnique.ALL_WITH_ALL
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=train_dataset.collate_fn)

    val_dataset = TripletDataset(
        file_path=path_args.dataset_dir / "val.csv",
        seq_coder=coder,
        max_per_group=max_per_group,
        random_size=1000
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=val_dataset.collate_fn)

    epochs = 5

    trainer_root_dir = path_args.artifacts_dir / "trainer"
    trainer_root_dir.mkdir(exist_ok=True, parents=True)
    trainer = L.Trainer(
        default_root_dir=trainer_root_dir,
        accelerator="gpu",
        devices="1",
        logger=[pl_loggers.MLFlowLogger(), pl_loggers.WandbLogger(f"{data_name} {path_to_save.parent.name}", project="similarity")],
        callbacks=callback_factory(callback_args),
        max_epochs=epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        log_every_n_steps=10,
        val_check_interval=50,
        # accumulate_grad_batches=5
    )
    cross_encoder = S3M(evaluating=False, vocab_size=len(coder), token_emb_dim=100, hidden_size=200, dropout=0.15).to(device)
    model = CrossEncoderModel(encoder=cross_encoder, scope=scope, loss=RankNetRerankerLoss(), train_size=len(train_dataset), batch_size=64, epochs=epochs)

    try:
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except KeyboardInterrupt as e:
        logger.error(f"Error: {e}")

    trainer.save_checkpoint(path_to_save)
    logger.info(f"Model saved to {path_to_save}")
    # sim_stack_model = SiamMultiModalModel(coder, encoder=model.encoder, similarity=model.similarity)
    # pair_stack_sim_model = PairStackBasedSimModel(data, sim_stack_model, MaxIssueScorer(), filter_model, index_model)
    # return pair_stack_sim_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(parents=[setup_model_parser(), setup_train_markup_parser(), setup_data_parser()])
    parser.add_argument("--path_to_save", type=Path, required=True)
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
    set_seed(args.seed or random_seed)

    data_segment = Segment(
        train=DataSegment(args.train_start, args.train_longitude),
        val=DataSegment(args.val_start, args.val_longitude),
        test=DataSegment(args.test_start, args.test_longitude)
    )

    path_args = PathArgs(
        actions_path=args.labels_dir / LabelFileNames.actions,
        state_path=args.labels_dir / LabelFileNames.state,
        reports_dir=args.reports_dir,
        dataset_dir=args.dataset_dir,
        artifacts_dir=args.artifacts_dir,
    )

    # index_args = IndexArgs(
    #     hyp_top_issues=args.hyp_top_issues,
    #     hyp_top_stacks=args.hyp_top_stacks,
    #     index_top_stacks=args.index_top_stacks
    # )

    callback_args = CallbackArgs(
        checkpoint_dir=args.path_to_save.parent,
    )

    train_similarity_model(
        data_name=args.data_name,
        scope=args.scope,
        data_segment=data_segment,
        path_args=path_args,
        callback_args=callback_args,
        forget_days=args.forget_days,
        max_per_group=args.max_per_group,
        path_to_save=args.path_to_save
    )

 
if __name__ == "__main__":
    run()
