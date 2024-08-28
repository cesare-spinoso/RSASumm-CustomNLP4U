import os
from pathlib import Path

# Repo specific constants
SRC_DIRECTORY = Path(__file__).parent

LOG_CONFIG_PATH = os.path.join(SRC_DIRECTORY, "..", "configs", "logging.conf")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

HF_TOKEN = "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"

DATASET_NAMES = {"covidet", "debatepedia", "duc_single", "multioped", "qmsum", "squality"}

METRICS_CACHE_DIR = os.path.join(SRC_DIRECTORY, "..", "data", "evaluation", "metrics_cache")

WANDB_DIRECTORY = os.path.join(SRC_DIRECTORY, "..", "data", "wandb")

# Mila cluster specific constants
SCRATCH_DIR = SRC_DIRECTORY.parent.parent / "scratch"

SCRATCH_CACHE_DIR = SCRATCH_DIR / ".cache"
