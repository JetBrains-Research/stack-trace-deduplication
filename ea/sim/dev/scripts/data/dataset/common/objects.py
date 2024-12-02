import typing as tp

from ea.sim.main.data.buckets.bucket_data import DataSegment
from ea.sim.main.utils import StackId


class ReportState(tp.NamedTuple):
    report_id: StackId
    issue_id: StackId

    columns = ["rid", "iid"]


class Dataset(tp.NamedTuple):
    train: list[ReportState]
    val: list[ReportState]
    test: list[ReportState]


class Segment(tp.NamedTuple):
    train: DataSegment
    val: DataSegment
    test: DataSegment
