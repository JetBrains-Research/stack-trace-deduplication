import json
import pickle
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class Writer:
    _instance: "Writer"

    def __init__(self, folder: Path):
        self.folder = folder
        self.folder.mkdir(exist_ok=True, parents=True)
        Writer.instance = self

    @staticmethod
    def file_name(prefix: str, ext: str) -> str:
        now = datetime.today().strftime('%Y-%m-%d-%H-%M')
        uid = uuid.uuid4().hex[:8]
        return f"{prefix}_{now}_{uid}.{ext}"

    def save_as_pickle(self, artifact: Any, prefix: str):
        name = self.file_name(prefix, "pkl")
        try:
            with (self.folder / name).open("wb") as file:
                pickle.dump(artifact, file)
            logger.debug(f"Saved '{name}' pkl artifact to {self.folder}")
        except Exception as e:
            logger.error(f"Failed to save '{name}' pkl artifact: '{e}'")

    def save_as_json(self, artifact: Any, prefix: str):
        name = self.file_name(prefix, "json")
        try:
            with (self.folder / name).open("w") as file:
                json.dump(artifact, file, indent=2)
            logger.debug(f"Saved '{name}' json artifact to {self.folder}")
        except Exception as e:
            logger.error(f"Failed to save '{name}' json artifact: '{e}'")

    @classmethod
    def initialize(cls, folder: Path):
        cls._instance = Writer(folder)

    @classmethod
    def get_instance(cls) -> "Writer":
        if cls._instance is None:
            raise ValueError("Writer is not initialized yet")
        return cls._instance
