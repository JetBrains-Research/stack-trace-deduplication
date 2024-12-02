from typing import Iterable

from ea.sim.main.data.buckets.event_state_model import StackAdditionEvent
from ea.sim.main.data.duplicates import HashStorage
from ea.sim.main.utils import StackId, IssueId


class StackStateModel:
    def __init__(self):
        self.stack_to_issue: dict[StackId, IssueId] = dict()
        self.stack_ids_seq: list[StackId] = []  # Received reports in chronological order
        self.action_to_index: dict[int, int] = dict()  # Chronological order of action
        # self.action_to_idx = {}

    def add_event(self, action: StackAdditionEvent):
        assert action.issue_id > 0, "Can't add detach event right now"
        self.stack_to_issue[action.stack_id] = action.issue_id
        self.action_to_index[action.id] = len(self.stack_ids_seq)
        self.stack_ids_seq.append(action.stack_id)

    def add(self, actions: Iterable[StackAdditionEvent]):
        for action in actions:
            self.add_event(action)

    def all_stacks(self, action_id: int, unique_across_issues: bool) -> list[int]:
        # "Unique_across_issues" means that from every issue will be fetched only stack ids without duplicates.
        # Duplicating based on hash.
        stacks_until_actions = self.stack_ids_seq[:self.action_to_index[action_id]]
        if not unique_across_issues:
            return stacks_until_actions

        hash_storage = HashStorage.get_instance()
        issue_to_hashes: dict[IssueId, set[int]] = dict()
        stacks_unique_across_issues: list[StackId] = []
        # Filtering stacks for uniqueness across issues.
        for stack_id in stacks_until_actions:
            hash = hash_storage.hash(stack_id)
            issue_id = self.stack_to_issue[stack_id]
            if issue_id not in issue_to_hashes:
                issue_to_hashes[issue_id] = set()

            # Stack is unique in the current issue (the same hash was not found).
            if hash not in issue_to_hashes[issue_id]:
                stacks_unique_across_issues.append(stack_id)
                issue_to_hashes[issue_id].add(hash)

        return stacks_unique_across_issues

