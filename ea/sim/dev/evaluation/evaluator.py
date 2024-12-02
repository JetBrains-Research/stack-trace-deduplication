import json
from pathlib import Path

from loguru import logger

from ea.sim.dev.evaluation.metrics.wrappers.attach_roc_auc import AttachRocAuc
from ea.sim.dev.evaluation import Prediction, EvaluationResult
from ea.sim.dev.evaluation.metrics import RetrievalAccuracy, RetrievalMRR, AttachFBeta, AttachFBetaV2, Metric


class Evaluator:
    metrics: list[Metric] = [
        RetrievalAccuracy(ks=[1, 3, 5, 10], boostrap=True),
        RetrievalMRR(boostrap=True),
        AttachFBeta(betas=[0.25, 0.5, 1, 2, 3]),
        AttachFBetaV2(betas=[0.25, 0.5, 1, 2, 3]),
        AttachRocAuc()
    ]

    save_file_name: str = "results.json"

    def __init__(self, predictions_path: Path, save_folder: Path):
        self._predictions_path = predictions_path
        self._save_folder = save_folder

    def _load_scores(self) -> list[Prediction]:
        predictions_jlist = json.loads(self._predictions_path.read_text())
        predictions = [Prediction.from_dict(prediction_jdict) for prediction_jdict in predictions_jlist]
        return predictions

    def _evaluate_metrics(self, preds: list[Prediction]) -> list[EvaluationResult]:
        results = [result for metric in self.metrics for result in metric(preds)]
        return results

    def _save_metrics(self, results: list[EvaluationResult]) -> None:
        self._save_folder.mkdir(parents=True, exist_ok=True)
        file_path = self._save_folder / self.save_file_name
        json_dump = json.dumps([result.to_dict() for result in results], indent=2)
        file_path.write_text(json_dump)

    def run(self):
        scores = self._load_scores()

        num_new_issues = sum(1 for score in scores if score.is_new_issue)
        num_attach_events = len(scores) - num_new_issues

        logger.info(f"New issues: {num_new_issues}, attach events: {num_attach_events}")

        metrics = self._evaluate_metrics(scores)
        self._save_metrics(metrics)
