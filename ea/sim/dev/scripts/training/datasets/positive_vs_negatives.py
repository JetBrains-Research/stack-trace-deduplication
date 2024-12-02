from pathlib import Path

from torch.utils.data import Dataset

from ea.sim.main.preprocess.seq_coder import SeqCoder


class PositiveSeveralNegativesDataset(Dataset):
    def __init__(self, file_path: Path, seq_coder: SeqCoder, random_size: int | None = None):
        raise NotImplementedError

    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
