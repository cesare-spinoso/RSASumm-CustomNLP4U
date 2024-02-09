import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from src import SRC_DIRECTORY
from src.utils.git import commit


def main_decorator(func):
    def wrapper(cfg: DictConfig):
        # Extract the config
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["finished_running"] = False
        # Pre-run commit
        run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        with open(
            os.path.join(cfg["write_config_directory"], f"{run_name}.yaml"), "w"
        ) as f:
            OmegaConf.save(cfg, f)
        commit(cwd=SRC_DIRECTORY, msg=f"Pre-run commit for : {run_name}")
        # Run the main function
        func(run_name, cfg)
        # Post-run commit
        cfg["finished_running"] = True
        with open(
            os.path.join(cfg["write_config_directory"], f"{run_name}.yaml"), "w"
        ) as f:
            OmegaConf.save(cfg, f)
        commit(cwd=SRC_DIRECTORY, msg=f"Post-run commit for : {run_name}")

    return wrapper
