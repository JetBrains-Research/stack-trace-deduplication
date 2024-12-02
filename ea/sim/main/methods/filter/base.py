from abc import ABC, abstractmethod

from ea.sim.main.data.objects.issue import Issue


class Filter(ABC):
    @abstractmethod
    def partial_fit(self, sim_train_data: list[tuple[int, int, int]], unsup_data: list[int] | None = None) -> 'Filter':
        raise NotImplementedError

    def refit(self) -> 'Filter':
        return self

    @abstractmethod
    def filter_top(self, event_id: int, stack_id: int, issues: dict[int, Issue]) -> dict[int, list[int]]:
        raise NotImplementedError
