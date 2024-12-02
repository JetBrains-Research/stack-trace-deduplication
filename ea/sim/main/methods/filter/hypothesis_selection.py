from ea.sim.main.data.objects.issue import Issue
from ea.sim.main.methods.base import SimStackModel
from ea.sim.main.methods.filter.base import Filter
from ea.sim.main.methods.issue_scorer import IssueScorer


class HypothesisSelector(Filter):
    def __init__(
            self,
            stack_model: SimStackModel,
            issue_scorer: IssueScorer,
            top_issues: int | None = None,
            top_stacks: int | None = None
    ):
        self.stack_model = stack_model
        self.issue_scorer = issue_scorer
        self.top_stacks = top_stacks or top_issues
        assert self.top_stacks is not None
        self.top_issues = top_issues
        self.cache = {}

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]],
            unsup_data: list[int] | None = None
    ) -> 'HypothesisSelector':
        self.stack_model.fit(sim_train_data, unsup_data)
        return self

    def filter_top(self, event_id: int, stack_id: int, issues: dict[int, Issue]) -> dict[int, list[int]]:
        cache_key = event_id, len(issues)
        if cache_key not in self.cache:
            res = []
            for id, issue in issues.items():
                stack_events = issue.stack_ids()
                if len(stack_events) == 0:
                    continue
                stacks = [event.stack_id for event in stack_events]
                preds = self.stack_model.predict(stack_id, stacks)
                res += list(zip([id for _ in stacks], stacks, preds))
            res = sorted(res, key=lambda x: -x[2])
            filtered_issues = {}
            stacks_cnt = 0
            for id, stack, pred in res:
                if id not in filtered_issues:
                    filtered_issues[id] = [stack]
                    stacks_cnt += 1
                elif (self.top_stacks is None) or (stacks_cnt < self.top_stacks):
                    filtered_issues[id].append(stack)
                    stacks_cnt += 1

                if (self.top_stacks is not None) and (self.top_issues is not None) \
                        and (stacks_cnt >= self.top_stacks) and (len(filtered_issues) >= self.top_issues):
                    break
            self.cache[cache_key] = filtered_issues
        return self.cache[cache_key]

    def __str__(self) -> str:
        return f"{self.stack_model.name()}_{self.issue_scorer.name()}_ts{self.top_stacks}_ti{self.top_issues}"
