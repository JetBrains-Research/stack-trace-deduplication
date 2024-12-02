
from loguru import logger

from ea.sim.dev.scripts.data.preprocess.common.artifacs import Artifact
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import PreprocessStep


class SelectIssuesWithMarkers(PreprocessStep):
    def __init__(self, max_marked_reports_per_issue: int | None = None):
        self._max_marked_reports_per_issue = max_marked_reports_per_issue

    def run(self, sources: SlowOpsSources, artifacts: list[Artifact]):
        markers = sources.markers.first
        state = sources.state.first
        state_with_markers = state[state["rid"].isin(markers["rid"])]
        if self._max_marked_reports_per_issue is None:
            marked_issue_ids = set(state_with_markers["iid"])
        else:
            state_with_markers = state_with_markers.groupby("iid")["rid"].apply(set)
            state_with_markers = state_with_markers[
                state_with_markers.apply(lambda x: len(x) <= self._max_marked_reports_per_issue)
            ]
            marked_issue_ids = set(state_with_markers.index)

        prev_state = sources.state.last
        next_state = prev_state[prev_state["iid"].isin(marked_issue_ids)]
        logger.debug(
            f"State transformed from {len(prev_state)} rows to {len(next_state)} rows "
            f"({next_state.rid.nunique()} unique reports, {next_state.iid.nunique()} unique issues)."
        )
        sources.state.add(next_state)

    @property
    def name(self) -> str:
        return "SelectIssuesWithMarkers"
