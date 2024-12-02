from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ea.sim.dev.scripts.data.preprocess.common.objects import Source, Sources
from ea.sim.dev.scripts.data.preprocess.so.utils import FileNames


@dataclass
class SlowOpsSources(Sources):
    state: Source
    markers: Source
    comments: Source

    @staticmethod
    def load(folder: Path) -> "SlowOpsSources":
        return SlowOpsSources(
            state=Source(pd.read_csv(folder / FileNames.state)),
            markers=Source(pd.read_csv(folder / FileNames.markers, sep="\t")),
            comments=Source(pd.read_csv(folder / FileNames.comments, sep="\t"))
        )

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)

        self.state.last.to_csv(folder / FileNames.state, index=False)
        self.markers.last.to_csv(folder / FileNames.markers, sep="\t", index=False)
        self.comments.last.to_csv(folder / FileNames.comments, sep="\t", index=False)
