from typing import Iterable

from ea.sim.dev.scripts.training.common.writer import Writer
from ea.sim.main.data.buckets.bucket_data import BucketData
from ea.sim.main.data.buckets.event_state_model import StackAdditionState
from ea.sim.main.data.objects.issue import Issue
from ea.sim.main.methods.base import SimIssueModel, SimStackModel
from ea.sim.main.methods.filter.base import Filter
from ea.sim.main.methods.index.base import Index
from ea.sim.main.methods.issue_scorer import IssueScorer
from ea.sim.main.utils import IssueId, StackId, Score


class PairStackBasedSimModel(SimIssueModel):
    min_score = float("-inf")

    def __init__(
            self,
            data: BucketData,
            stack_model: SimStackModel,
            issue_scorer: IssueScorer,
            filter_model: Filter | None = None,
            index_model: Index | None = None,
            verbose: bool = True,
    ):
        self.data = data  # Remove in the future
        self.stack_model = stack_model
        self.issue_scorer = issue_scorer
        self.filter_model = filter_model
        self.index_model = index_model

        self.verbose = verbose

        assert not (self.filter_model is not None and self.index_model is not None)

    def partial_fit(
            self,
            sim_train_data: list[tuple[int, int, int]],
            unsup_data: list[int] | None = None
    ) -> 'PairStackBasedSimModel':
        self.filter_model.partial_fit(sim_train_data, unsup_data)
        self.stack_model.fit(sim_train_data, unsup_data)
        return self

    def predict_all(
            self,
            stack_id: StackId,
            issues: dict[IssueId, Issue],
            with_stacks: bool = False
    ) -> tuple[dict[IssueId, Score | tuple[Score, StackId]], int]:
        pred_issues = {}
        stacks_cnt = 0
        for issue_id, issue in issues.items():
            stack_ids = issue.stack_ids()
            stacks_cnt += len(stack_ids)
            if len(stack_ids) == 0:
                if with_stacks:
                    pred_issues[issue_id] = self.min_score, -1
                else:
                    pred_issues[issue_id] = self.min_score
                continue
            preds = self.stack_model.predict(stack_id, stack_ids)
            score = self.issue_scorer.score(preds, with_arg=with_stacks)
            if with_stacks:
                score, ind = score
                pred_issues[issue_id] = score, stack_ids[ind]
            else:
                pred_issues[issue_id] = score
        return pred_issues, stacks_cnt

    def predict_with_index(
            self,
            event_id: int,
            stack_id: StackId,
            issues: dict[IssueId, Issue],
            with_stacks: bool = False
    ) -> dict[IssueId, Score | tuple[Score, StackId]]:
        pred_issues = {issue_id: self.min_score for issue_id in issues}
        stack_ids_for_index = self.data.all_reports_until_event(
            event_id,
            unique_across_issues=self.issue_scorer.support_dup_removal
        )
        self.index_model.insert(stack_ids_for_index)
        top_issues = self.index_model.fetch_top(event_id, stack_id, issues, with_scores=True)
        for issue_id, stacks_zip_scores in top_issues.items():
            stack_ids, scores = zip(*stacks_zip_scores)
            score = self.issue_scorer.score(scores, with_arg=with_stacks)
            if with_stacks:
                score, ind = score
                pred_issues[issue_id] = score, stack_ids[ind]
            else:
                pred_issues[issue_id] = score
        return pred_issues

    def predict_with_filter(
            self,
            event_id: int,
            stack_id: int,
            issues: dict[int, Issue],
            with_stacks: bool = False
    ) -> tuple[dict[IssueId, Score | tuple[Score, StackId]], int]:
        pred_issues = {}
        stacks_cnt = 0
        top_issues = self.filter_model.filter_top(event_id, stack_id, issues)
        for issue_id, stack_ids in top_issues.items():
            preds = self.stack_model.predict(stack_id, stack_ids)
            score = self.issue_scorer.score(preds, with_arg=with_stacks)
            if with_stacks:
                score, ind = score
                pred_issues[issue_id] = score, stack_ids[ind]
            else:
                pred_issues[issue_id] = score
            stacks_cnt += len(stack_ids)

        # save other issues for fair map and other metrics comparison
        for is_id, issue in issues.items():
            if is_id not in top_issues:
                if with_stacks:
                    pred_issues[is_id] = self.min_score, issue.stack_ids()[0]
                else:
                    pred_issues[is_id] = self.min_score
        return pred_issues, stacks_cnt

    def predict(
            self,
            events: Iterable[StackAdditionState]
    ) -> Iterable[tuple[IssueId, dict[IssueId, Score | tuple[Score, StackId]]]]:
        writer = Writer.get_instance()

        for event in events:
            if self.index_model:
                pred_issues = self.predict_with_index(event.id, event.stack_id, event.issues)
                writer.save_as_json(pred_issues, f"predict_index_event={event.id}_stack_id={event.stack_id}")
            elif self.filter_model:
                pred_issues, _ = self.predict_with_filter(event.id, event.stack_id, event.issues)
            else:
                pred_issues, _ = self.predict_all(event.stack_id, event.issues)
            yield event.issue_id, pred_issues

    def predict_debug(self, events: Iterable[StackAdditionState]) -> list[tuple[int, dict[int, float]]]:
        res = []
        correct_cnt, incorrect_cnt = 0, 0
        for i, event in enumerate(events):
            if self.filter_model:
                pred_issues, _ = self.predict_with_filter(event.id, event.stack_id, event.issues)
                if event.issue_id not in pred_issues or max(pred_issues.values()) != pred_issues[event.issue_id]:
                    max_score = max(pred_issues.values())
                    max_score_id = [id for id in pred_issues if pred_issues[id] == max_score][0]
                    selected_issues = {max_score_id: event.issues[max_score_id],
                                       event.issue_id: event.issues[event.issue_id]}

                    pred_issues_with_stack, _ = self.predict_with_filter(event.id, event.stack_id, selected_issues,
                                                                         with_stacks=True)
                    initial_stack = event.stack_id
                    target_score, target_stack = pred_issues_with_stack[event.stack_id]
                    lerch_score, lerch_stack = pred_issues_with_stack[max_score_id]
                    # {"stack": self.stack_model.coder.stack_loader(initial_stack), "target": self.stack_model.coder.stack_loader(target_stack), "lerch": self.stack_model.coder.stack_loader(lerch_stack)}

                    incorrect_cnt += 1
                else:
                    correct_cnt += 1
            else:
                pred_issues, _ = self.predict_all(event.stack_id, event.issues)
            res.append((event.issue_id, pred_issues))
        return res

    def predict_at_hyp(self, events: Iterable[StackAdditionState]) -> list[tuple[int, dict[int, float]]]:
        res = []
        for i, event in enumerate(events):
            if self.filter_model:
                pred_issues = {}
                issues = self.filter_model.filter_top(event.id, event.stack_id, event.issues)
                for id in issues:
                    if id == event.issue_id:
                        pred_issues[id] = 1.
                    else:
                        pred_issues[id] = 0.
            else:
                raise ValueError("filter model is not defined")
            res.append((event.issue_id, pred_issues))
        return res

    def predict_wo_new(self, events: Iterable[StackAdditionState], known_issues_ids: set[int]) \
            -> list[tuple[int, dict[int, float]]]:
        res = []
        cnt = 0
        for i, event in enumerate(events):
            if event.issue_id not in event.issues:  # event.st_id in event.issues[event.is_id].stacks == True  # len(event.issues[event.is_id].stacks) < 2
                cnt += 1
                continue
            if self.filter_model:
                pred_issues, _ = self.predict_filtered(event.id, event.stack_id, event.issues)
            else:
                pred_issues, _ = self.predict_all(event.stack_id, event.issues)
            res.append((event.issue_id, pred_issues))
        print("filter", cnt, len(res))
        return res

    def name(self) -> str:
        return "_".join([model.name() for model in [self.stack_model, self.issue_scorer, self.filter_model]])

    # def predict_if_in_hyp(self, events: Iterable[StackAdditionState]) -> List[Tuple[int, dict[int, float]]]:
    #     res = []
    #     for i, event in enumerate(events):
    #         if self.filter_model:
    #             pred_issues = {}
    #             issues = self.filter_model.filter_top(event.id, event.stack, event.issues)
    #             if event.is_id in issues:
    #                 for id, stack_ids in issues.items():
    #                     preds = self.stack_model.predict(event.stack, stack_ids)
    #                     pred_issues[id] = self.issue_scorer.score(preds)
    #                 res.append((event.is_id, pred_issues))
    #             else:
    #                 print("a")
    #         else:
    #             raise ValueError("filter model is not defined")
    #     return res  # 9 / 140, 92 / 1106 всего

    # def analyze_eq_score(self, events: Iterable[StackAdditionState], coder: SeqCoder):
    #     eq_cases_num = 0
    #     all_eq_cases_num = 0
    #     events_num = 0
    #     for i, event in enumerate(events):
    #         if self.filter_model is not None:
    #             pred_issues, _ = self.predict_filtered(event.id, event.stack, event.issues)
    #         else:
    #             pred_issues, _ = self.predict_all(event.stack, event.issues)
    #         max_value = max(pred_issues.values())
    #         max_keys = set(k for k, v in pred_issues.items() if max_value == v)
    #
    #         events_num += 1
    #         if len(max_keys) > 1 and event.is_id in max_keys:
    #             eq_cases_num += 1
    #             stack_pred_issues, _ = self.predict_filtered(event.id, event.stack, event.issues, with_stacks=True)
    #             max_stacks_ids = [v[1] for k, v in stack_pred_issues.items() if k in max_keys]
    #             max_stacks_frames = [tuple(coder(stack_id)) for stack_id in max_stacks_ids]
    #             set_max_stacks_frames = set(max_stacks_frames)
    #             if len(set_max_stacks_frames) == 1:
    #                 all_eq_cases_num += 1
    #             else:
    #                 a = 0
    #     if eq_cases_num > 0:
    #         print("Eq unique ratio", all_eq_cases_num / eq_cases_num, "Eq max ratio", eq_cases_num / events_num)
