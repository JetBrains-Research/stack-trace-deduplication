import json

from loguru import logger

from ea.sim.main.data.stack_loader import StackLoader
from ea.sim.main.utils import StackId, ARTIFACTS_DIR


class HashStorage:
    """
    Stores hashes for all stacks for finding duplicates.
    If two stacks have the same hashes, then they are duplicates.
    """
    _instance = None
    _file_name = "hashes.json"

    def __init__(self, stack_loader: StackLoader):
        self._stack_loader = stack_loader
        self._hashes: dict[StackId, int] = dict()

    def hash(self, stack_id: StackId) -> int:
        if stack_id not in self._hashes:
            self._hashes[stack_id] = hash(self._stack_loader(stack_id))
        return self._hashes[stack_id]

    def hashes(self, stack_ids: list[StackId]) -> list[int]:
        return [self.hash(stack_id) for stack_id in stack_ids]

    @classmethod
    def initialize(cls, stack_loader: StackLoader):
        cls._instance = HashStorage(stack_loader).load()

    @classmethod
    def get_instance(cls) -> "HashStorage":
        if cls._instance is None:
            raise ValueError("HashStorage not initialized yet")
        return cls._instance

    def load(self) -> "HashStorage":
        file_path = ARTIFACTS_DIR / HashStorage._file_name
        if file_path.exists():
            hashes_jdict = json.loads(file_path.read_text())
            self._hashes = {int(stack_id): hash for (stack_id, hash) in hashes_jdict.items()}
            logger.debug(f"Loaded hash storage state from '{file_path}'")
        else:
            logger.debug("Could not find HashStorage state, creating scratch")
        return self

    @staticmethod
    def has_fitted() -> bool:
        file_path = ARTIFACTS_DIR / HashStorage._file_name
        return file_path.exists()

    def save(self):
        hashes_dump = json.dumps(self._hashes, indent=2, sort_keys=True)
        save_path = ARTIFACTS_DIR / HashStorage._file_name
        save_path.write_text(hashes_dump)
        logger.debug(f"HashStorage saved to '{save_path}'")
