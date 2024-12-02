import os
from enum import Enum
from pathlib import Path
from typing import TypeAlias

import torch

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
#     device = torch.device("mps")

artifacts_dir_key = "ARTIFACTS_DIR"
if artifacts_dir_key in os.environ:
    ARTIFACTS_DIR = Path(os.environ[artifacts_dir_key]).expanduser().resolve()
else:
    ARTIFACTS_DIR = Path.home() / "artifacts" / "sim"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# Types
StackId: TypeAlias = int
IssueId: TypeAlias = int
Score: TypeAlias = float
Timestamp: TypeAlias = int


class Timestamps:
    YEAR_2020 = 1_577_833_200_000
    YEAR_2021 = 1_609_455_600_000
    YEAR_2022 = 1_640_991_600_000


class Scope(str, Enum):
    MainProject = "main_project"
    SlowOps = "slowops"
    SideProject = "side_project"
    NetBeans = "netbeans"
    Combined = "combined"
    Eclipse = "eclipse"
    Campbell = "campbell"
    Gnome = "gnome"


class Method(str, Enum):
    # Neural
    S3M = "s3m"
    NeuralLerch = "neural_lerch"

    # Classic
    Lerch = "lerch"
    Cosine = "cosine"
    Prefix = "prefix"
    ReBucket = "rebucket"
    TraceSim = "tracesim"
    Levenshtein = "levenshtein"
    Brodie = "brodie"
    Moroo = "moroo"
    Irving = "irving"
    Durfex = "durfex"
    Linreg = "linreg"
    CrashGraphs = "crash_graphs"
    LSI = "lsi"
    FaST = "fast"
