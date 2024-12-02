from abc import ABC, abstractmethod

from ea.sim.main.data.buckets.event_state_model import StackAdditionEvent
from ea.sim.main.utils import Timestamps, Scope


class InvalidDatasetError(ValueError):
    pass


class EventsData(ABC):
    day_secs = 60 * 60 * 24

    @abstractmethod
    def events(self) -> list[StackAdditionEvent]:
        raise NotImplementedError

    @staticmethod
    def start_timestamp(scope: Scope) -> int | None:
        if scope == Scope.MainProject:
            return Timestamps.YEAR_2022
        else:
            return None
