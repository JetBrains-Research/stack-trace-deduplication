import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import torch
import pytorch_lightning as pl

random_seed = 5


def set_seed(seed: int = random_seed):
    """
    https://pytorch.org/docs/stable/notes/randomness.html
    https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=False)
    # torch._set_deterministic(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def save_pickle(obj: Any, path: Union[str, Path]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as file:
        return pickle.load(file)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def save_json(obj: Any, path: Union[str, Path]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(obj, file, indent=2)

# def save_to_artifact(obj: Any, path: Any, save_func):
#     with tc.publishArtifact(path) as artifact_file:
#         save_func(obj, artifact_file)
