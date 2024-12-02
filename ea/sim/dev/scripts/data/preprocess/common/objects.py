from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class Source:
    def __init__(self, df: pd.DataFrame):
        self.transforms: list[pd.DataFrame] = [
            df
        ]

    def add(self, df: pd.DataFrame):
        self.transforms.append(df)

    @property
    def first(self) -> pd.DataFrame:
        return self.transforms[0]

    @property
    def last(self) -> pd.DataFrame:
        return self.transforms[-1]


@dataclass
class Sources(ABC):
    @staticmethod
    @abstractmethod
    def load(folder: Path) -> "Sources":
        raise NotImplementedError

    @abstractmethod
    def save(self, folder: Path):
        raise NotImplementedError
