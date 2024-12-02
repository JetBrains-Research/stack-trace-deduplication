from abc import ABC, abstractmethod

from loguru import logger

from ea.sim.main.data.objects.issue import Issue


class IssuesSelector(ABC):
    def __init__(self, forget_days: int | None = None):
        self.forget_days = forget_days

    @abstractmethod
    def select(self, issues: dict[int, Issue], current_day: int) -> dict[int, Issue]:
        raise NotImplementedError


class LastUpdateIssueSelector(IssuesSelector):
    def __init__(self, forget_days: int | None = None):
        super().__init__(forget_days)

    def select(self, issues: dict[int, Issue], current_day: int) -> dict[int, Issue]:
        if self.forget_days is None:
            return issues

        return {
            issue_id: issue
            for issue_id, issue in issues.items() if (current_day - issue.last_ts) < self.forget_days
        }


class ReportTimeIssueSelector(IssuesSelector):
    def __init__(self, forget_days: int | None = None):
        super().__init__(forget_days)
        logger.warning("Used 'ReportTimeIssueSelector', might be slow")

    def create_sliced_issue(self, issue: Issue, current_day: int) -> Issue | None:
        time_sliced_events = {
            id: event
            for id, event in issue.stack_events.items() if (current_day - event.ts) < self.forget_days
        }
        return Issue.from_events(issue.id, time_sliced_events) if len(time_sliced_events) > 0 else None

    def select(self, issues: dict[int, Issue], current_day: int) -> dict[int, Issue]:
        if self.forget_days is None:
            return issues

        issues = {issue.id: self.create_sliced_issue(issue, current_day) for issue in issues.values()}
        return {issue_id: issue for issue_id, issue in issues.items() if issue is not None}
