from pathlib import Path
from typing import Iterable
import time

from ea.sim.dev.evaluation import Prediction, ScoreRecord
from ea.sim.main.data.buckets.bucket_data import BucketData
from ea.sim.main.data.buckets.event_state_model import StackAdditionState
from ea.sim.main.data.objects.issue import Issue
from ea.sim.main.methods.issue_scorer import IssueScorer
from ea.sim.main.methods.retrieval_model import RetrievalModel, CachedRetrievalModel
from ea.sim.main.methods.scoring_model import ScoringModel, CachedScoringModel
from ea.sim.main.utils import StackId, IssueId


class RankingModel:
    def __init__(
            self,
            data: BucketData,
            retrieval_model: RetrievalModel,
            scoring_model: ScoringModel,
            issue_scorer: IssueScorer,
    ):
        self._data = data
        # self._retrieval_model = CachedRetrievalModel(retrieval_model)
        # self._scoring_model = CachedScoringModel(scoring_model)
        self._retrieval_model = retrieval_model
        self._scoring_model = scoring_model
        self._issue_scorer = issue_scorer

        self.retrieval_times = []
        self.scoring_times = []

    def get_average_times(self) -> tuple[float, float]:
        retrieval_time = sum(self.retrieval_times) / len(self.retrieval_times)
        scoring_time = sum(self.scoring_times) / len(self.scoring_times)
        return retrieval_time, scoring_time

    def get_candidates(self, issues: dict[IssueId, Issue]) -> dict[StackId, IssueId]:
        return {
            stack_id: issue.id
            for issue in issues.values()
            for stack_id in issue.stack_ids(unique=True)
        }

    def predict_stack_scores(self, anchor_id: StackId, candidate_ids: list[StackId]) -> list[ScoreRecord]:
        time_start = time.time()
        candidate_ids = self._retrieval_model.search(anchor_id, candidate_ids)
        self.retrieval_times.append(time.time() - time_start)
        scores = self._scoring_model.predict(anchor_id, candidate_ids)
        self.scoring_times.append(time.time() - time_start)
        score_records = [ScoreRecord(candidate_id, score) for candidate_id, score in zip(candidate_ids, scores)]
        score_records = sorted(score_records, key=lambda x: x.score, reverse=True)
        return score_records

    def predict_issue_scores(
            self,
            candidates: dict[StackId, IssueId],
            stack_scores: list[ScoreRecord],
    ) -> list[ScoreRecord]:
        issue_all_scores = {}
        for stack_record in stack_scores:
            issue_id = candidates[stack_record.object_id]
            if issue_id not in issue_all_scores:
                issue_all_scores[issue_id] = []
            issue_all_scores[issue_id].append(stack_record.score)

        issue_scores = {}
        issue_candidate_ids = candidates.values()
        for issue_id in issue_candidate_ids:
            issue_scores[issue_id] = self.min_score
            if issue_id in issue_all_scores:
                issue_scores[issue_id] = self._issue_scorer.score(issue_all_scores[issue_id])
        issue_scores = [ScoreRecord(issue_id, score) for issue_id, score in issue_scores.items()]
        issue_scores = sorted(issue_scores, key=lambda x: x.score, reverse=True)
        return issue_scores

    def predict(self, events: Iterable[StackAdditionState]) -> Iterable[Prediction]:
        for event in events:
            candidates = self.get_candidates(event.issues)
            stack_scores = self.predict_stack_scores(event.stack_id, list(candidates.keys()))
            issue_scores = self.predict_issue_scores(candidates, stack_scores)
            yield Prediction(
                query_id=event.stack_id,
                target_id=event.issue_id,
                stack_scores=stack_scores,
                issue_scores=issue_scores,
                is_new_issue=event.is_new_issue
            )

    def save_retrieval_cache(self, folder: Path) -> None:
        self._retrieval_model.save_cache(folder)

    def save_scoring_cache(self, folder: Path) -> None:
        self._scoring_model.save_cache(folder)

    def load_retrieval_cache(self, folder: Path) -> None:
        self._retrieval_model.load_cache(folder)

    @property
    def min_score(self) -> float:
        return self._scoring_model.min_score
