from dataclasses import asdict

from ea.sim.dev.scripts.data.dataset.common.collect_dataset import set_seed, Dataset, save
from ea.sim.dev.scripts.data.dataset.common.config import DatasetConfig, Paths
from ea.sim.dev.scripts.data.dataset.common.constraints import Constraints
from ea.sim.dev.scripts.data.dataset.common.miners import FinalStateReportMiner
from ea.sim.main.data.buckets.bucket_data import EventsBucketData
from ea.sim.main.data.buckets.events_extractors import EventsFromState
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.data.stack_loader import DirectoryStackLoader


def parse(config: DatasetConfig, paths: Paths):
    print("Config:", asdict(config))
    set_seed(config.random_seed)
    stack_loader = DirectoryStackLoader(paths.reports_dir)
    HashStorage.initialize(stack_loader)
    data = EventsBucketData(
        name=config.scope.value,
        events_data=EventsFromState(
            state_path=paths.state_path,
            labels_source=True,
            existed_reports=stack_loader.sid_to_dir.keys(),
            sec_denom=1_000,
            scope=config.scope
        ),
        stack_loader=stack_loader,
        forget_days=None
    ).load()

    miner = FinalStateReportMiner(data, Constraints.load())
    dataset = Dataset(
        train=list(miner.mine(config.segments.train)),
        val=list(miner.mine(config.segments.val, prev_segments=[config.segments.train])),
        test=list(miner.mine(config.segments.test, prev_segments=[config.segments.train])),
    )

    save(paths.save_dir, dataset, config)
