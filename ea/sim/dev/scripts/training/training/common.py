from pathlib import Path

from loguru import logger

from ea.sim.dev.scripts.data.dataset.common.objects import Segment
from ea.sim.main.data.buckets.bucket_data import BucketData, DataSegment, EventsBucketData
from ea.sim.main.data.buckets.events_extractors import EventsFromState
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.data.stack_loader import DirectoryStackLoader
from ea.sim.main.methods.index import FAISS, Index
from ea.sim.main.methods.neural.encoders.texts import Encoder
from ea.sim.main.utils import Scope


class LabelFileNames:
    actions: str = "actions.csv"
    state: str = "state.csv"


def fit_hash_storage(data: BucketData, segment: Segment):
    if HashStorage.has_fitted():
        logger.debug("HashStorage is already fitted, fitting is skipped")
    else:
        logger.debug("HashStorage is not fitted yet, starting to fit...")
        start_day = segment.train.start
        end_day = segment.test.start + segment.test.longitude
        longitude = end_day - start_day
        all_stacks_segment = DataSegment(start_day, longitude)

        all_stack_ids = data.all_reports(all_stacks_segment, unique_across_issues=False)
        hash_storage = HashStorage.get_instance()
        _ = hash_storage.hashes(all_stack_ids)
        hash_storage.save()


def create_bucket_data(
        data_name: str, scope: Scope,
        reports_dir: Path, actions_path: Path, state_path: Path,
        forget_days: int | None
) -> BucketData:
    stack_loader = DirectoryStackLoader(reports_dir)
    events_data = EventsFromState(
        state_path, actions_path, stack_loader.sid_to_dir.keys(), sec_denom=1_000, scope=scope
    )

    return EventsBucketData(
        name=data_name,
        events_data=events_data,
        stack_loader=stack_loader,
        forget_days=forget_days,
        sep=".",
        verbose=True
    ).load()


def create_index_model(encoder: Encoder) -> Index:
    index_model = FAISS(encoder)
    logger.debug(f"Created FAISS index model")
    return index_model
