import os
from pathlib import Path

# Repo paths
SRC_DIRECTORY = Path(__file__).parent

LOG_CONFIG_PATH = os.path.join(SRC_DIRECTORY, "..", "configs", "logging.conf")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Mila cluster specific constants
SCRATCH_DIR = SRC_DIRECTORY.parent.parent / "scratch"

SCRATCH_CACHE_DIR = SCRATCH_DIR / ".cache"
