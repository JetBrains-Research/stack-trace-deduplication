from itertools import combinations
from loguru import logger

from ea.sim.dev.scripts.data.preprocess.common.artifacs import Artifact, JsonArtifact
from ea.sim.dev.scripts.data.preprocess.so.markers import is_service_marker
from ea.sim.dev.scripts.data.preprocess.so.sources import SlowOpsSources
from ea.sim.dev.scripts.data.preprocess.so.steps import PreprocessStep


def are_duplicates(
        issue_1: set[int], issue_2: set[int],
        markers: dict[int, set[str]]
) -> bool:
    markers_1 = {rid_1: markers[rid_1] for rid_1 in issue_1 if rid_1 in markers}
    markers_2 = {rid_2: markers[rid_2] for rid_2 in issue_2 if rid_2 in markers}

    if (len(markers_1) == 0) or (len(markers_2) == 0):
        return False

    for rid_1, rid_markers_1 in markers_1.items():
        for rid_2, rid_markers_2 in markers_2.items():
            if rid_markers_1 == rid_markers_2:
                return True

    return False


class MergeIssuesDuplicates(PreprocessStep):
    def run(self, sources: SlowOpsSources, artifacts: list[Artifact]):
        state = sources.state.last
        issues = state.groupby("iid")["rid"].apply(set).to_dict()  # issue_id -> report_ids

        markers = sources.markers.first
        markers = markers[~markers["method_name"].isnull()]
        markers = markers[markers["rid"].isin(state["rid"])]
        markers = markers.groupby("rid")["method_name"].apply(set).to_dict()  # report_id -> markers
        markers = {
            rid: {name for name in names if not is_service_marker(name)}
            for rid, names in markers.items()
        }

        duplicates = {}
        issue_ids = sorted(issues.keys())

        combs = list(combinations(issue_ids, 2))
        for (iid_1, iid_2) in combs:
            if are_duplicates(issues[iid_1], issues[iid_2], markers):
                duplicates[max(iid_1, iid_2)] = min(iid_1, iid_2)

        # Solve chains (iid_1 replaced to iid_2, iid_2 replaced to iid_3 => iid_1 replaced to iid_3).
        for iid_1, iid_2 in duplicates.items():
            curr_iid_2 = iid_2
            next_iid_2 = duplicates.get(curr_iid_2, None)
            while next_iid_2 is not None:
                curr_iid_2 = next_iid_2
                next_iid_2 = duplicates.get(curr_iid_2, None)
            duplicates[iid_1] = curr_iid_2

        next_state = sources.state.last.copy()
        next_state["iid"] = next_state["iid"].replace(duplicates)
        logger.debug(f"Found {len(duplicates)} duplicated issues")
        sources.state.add(next_state)
        artifacts.append(JsonArtifact(duplicates, "issues_duplicates"))

    @property
    def name(self) -> str:
        return "MergeIssuesDuplicates"
