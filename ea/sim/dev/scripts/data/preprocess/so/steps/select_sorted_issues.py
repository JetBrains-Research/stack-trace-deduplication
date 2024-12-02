from loguru import logger

from ea.sim.dev.scripts.data.preprocess.common.artifacs import Artifact
from ea.sim.dev.scripts.data.preprocess.so.comments import is_sorted_comment
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import PreprocessStep


class SelectSortedIssuesStep(PreprocessStep):
    def run(self, sources: SlowOpsSources, artifacts: list[Artifact]):
        comments = sources.comments.first
        sorted_comments = comments[comments["comments"].apply(is_sorted_comment)]

        prev_state = sources.state.last
        next_state = prev_state[prev_state["iid"].isin(sorted_comments["iid"])]
        logger.debug(
            f"State transformed from {len(prev_state)} rows to {len(next_state)} rows "
            f"({next_state.rid.nunique()} unique reports, {next_state.iid.nunique()} unique issues)."
        )
        sources.state.add(next_state)

    @property
    def name(self) -> str:
        return "SelectSortedIssues"
