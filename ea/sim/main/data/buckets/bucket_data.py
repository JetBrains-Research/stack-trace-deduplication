from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, NamedTuple

from loguru import logger

from ea.sim.main.data.buckets.event_state_model import EventStateModel, StackAdditionEvent, StackAdditionState
from ea.sim.main.data.buckets.events_extractors import EventsData
from ea.sim.main.data.buckets.stack_state_model import StackStateModel
from ea.sim.main.data.stack_loader import StackLoader
from ea.sim.main.utils import StackId


class DataSegment(NamedTuple):
    start: int  # maybe negative the same as list indexes
    longitude: int


class BucketData(ABC):
    def __init__(
            self,
            name: str,
            stack_loader: StackLoader,
            events: list[StackAdditionEvent],
            event_state_model: EventStateModel,
            sep: str = ".",
            verbose: bool = False
    ):
        self.name = name
        self.stack_loader = stack_loader
        self.events = events
        self._all_reports: dict[tuple[int, int, bool], list[StackId]] = {}

        self.event_state_model = event_state_model
        self.stack_state_model = StackStateModel()
        self.sep = sep
        self.verbose = verbose

    @abstractmethod
    def load(self) -> 'BucketData':
        raise NotImplementedError

    def _time_slice_events(self, start: int, finish: int) -> list[StackAdditionEvent]:
        return [
            event for event in self.events
            if start <= event.ts < finish
        ]

    def _cached_event_state(self, until_day: int) -> EventStateModel:
        event_model = self.event_state_model

        if event_model.file_path(until_day).exists():
            event_model.load(until_day)
            logger.debug(f"EventModel loaded from file")
        else:
            load_prev = False
            for i in range(int(until_day), 0, -1):
                if event_model.file_path(i).exists():
                    logger.debug(f"Loaded from {event_model.file_path(i)}")
                    event_model.load(i)
                    event_model.warmup(self._time_slice_events(i, until_day))
                    load_prev = True
                    logger.debug(f"Post train from {i} to {until_day}")
                    break
            if not load_prev:
                event_model.warmup(self._time_slice_events(0, until_day))
            event_model.save(until_day)

        return event_model

    def _generate_events(
            self,
            start: int, longitude: int, *,
            only_labeled: bool, all_issues: bool, with_dup_attach: bool
    ) -> Iterable[StackAdditionState]:
        assert longitude >= 0
        if start < 0:  # from tail
            events_last_day = self.events[-1]
            start += events_last_day.ts

        event_model = self._cached_event_state(start)
        time_sliced_events = self._time_slice_events(start, start + longitude)
        return event_model.collect(
            time_sliced_events,
            only_labeled=only_labeled,
            all_issues=all_issues,
            with_dup_attach=with_dup_attach
        )

    def all_reports(self, data_segment: DataSegment, unique_across_issues: bool = False) -> list[int]:
        start = data_segment.start
        longitude = data_segment.longitude
        key = (start, longitude, unique_across_issues)
        if key not in self._all_reports:
            event_model = self._cached_event_state(start + longitude)
            seen_stacks = event_model.all_seen_stacks(start, start + longitude, only_unique_from_issue=unique_across_issues)
            self._all_reports[key] = sorted(seen_stacks)
        return self._all_reports[key]

    def all_reports_until_event(self, action_id: int, unique_across_issues: bool = False) -> list[int]:
        return self.stack_state_model.all_stacks(action_id, unique_across_issues)

    def warmed_up_event_model(self, until_day: int) -> EventStateModel:
        event_model = self._cached_event_state(until_day)
        return event_model

    def get_events(
            self,
            data_segment: DataSegment,
            *,
            only_labeled: bool = False,
            all_issues: bool = True,
            with_dup_attach: bool = True
    ) -> Iterable[StackAdditionState]:
        start = data_segment.start
        longitude = data_segment.longitude
        return self._generate_events(
            start,
            longitude,
            only_labeled=only_labeled,
            all_issues=all_issues,
            with_dup_attach=with_dup_attach
        )


class EventsBucketData(BucketData, ABC):
    def __init__(
            self,
            name: str,
            events_data: EventsData,
            stack_loader: StackLoader,
            forget_days: int | None = None,
            sep: str = ".",
            verbose: bool = False
    ):
        super().__init__(
            name=name,
            stack_loader=stack_loader,
            events=[],
            event_state_model=EventStateModel(name, forget_days, verbose),
            sep=sep,
            verbose=verbose
        )
        self.events_data = events_data

    def load(self) -> 'BucketData':
        self.events = self.events_data.events()
        logger.debug(f"Loaded {len(self.events)} bucket events day {self.events[0].ts} until {self.events[-1].ts}")
        self.stack_state_model.add(self.events)
        return self
