import re
from pathlib import Path
from typing import Callable

import pandas as pd

from ea.sim.main.data.buckets.event_state_model import StackAdditionEvent


class IssuesFilter:
    @staticmethod
    def with_comments(issue_to_comment: dict[int, str | None]) -> set[int]:
        return {
            iid for iid, comment in issue_to_comment.items()
            if isinstance(comment, str) and ("unsorted" not in comment.lower())
        }

    @staticmethod
    def with_tickets(issue_to_comment: dict[int, str | None]) -> set[int]:
        ticket_reg_exp = re.compile("[A-Z]+-[0-9]+")
        return {
            iid for iid, comment in issue_to_comment.items()
            if isinstance(comment, str) and ticket_reg_exp.search(comment)
        }


def load_sorted_issues(
        issues_path: Path,
        issue_filter: Callable[[dict[int, str]], set[int]] | None = None,
        delimiter: str = "\t"
) -> set[int]:
    issues_df = pd.read_csv(issues_path, delimiter=delimiter)
    if issue_filter is None:
        return set(issues_df.iid)

    issue_to_comment = dict(zip(issues_df.iid, issues_df.comments))
    return issue_filter(issue_to_comment)


def filter_actions(actions: list[StackAdditionEvent], sorted_issues: set[int]) -> list[StackAdditionEvent]:
    sid_to_iid = {}

    sorted_actions = []
    for action in actions:
        if action.issue_id == -1:
            if (action.stack_id in sid_to_iid) and (sid_to_iid[action.stack_id] in sorted_issues):
                sorted_actions.append(action)
        else:
            if action.issue_id in sorted_issues:
                sorted_actions.append(action)

        sid_to_iid[action.stack_id] = action.issue_id

    return sorted_actions
