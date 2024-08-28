import os
from pathlib import Path

# Repo specific constants
SRC_DIRECTORY = Path(__file__).parent

LOG_CONFIG_PATH = os.path.join(SRC_DIRECTORY, "XXX")

OPENAI_API_KEY = os.getenv("XXX")

HF_TOKEN = "XXX"

DATASET_NAMES = {"XXX"}

METRICS_CACHE_DIR = os.path.join(SRC_DIRECTORY, "XXX")

WANDB_DIRECTORY = os.path.join(SRC_DIRECTORY, "XXX")

SCRATCH_DIR = SRC_DIRECTORY.parent.parent / "XXX"

SCRATCH_CACHE_DIR = SCRATCH_DIR / "XXX"
