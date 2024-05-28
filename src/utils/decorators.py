import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

from src import SRC_DIRECTORY
from src.utils.git import commit

def test_decorator(func):
    def wrapper(cfg: DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
        print(datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f"))
        func(cfg)
    return wrapper

def main_decorator(func):
    def wrapper(cfg: DictConfig):
        # Extract the config
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["finished_running"] = False
        # Pre-run commit
        run_name = datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
        print(f"Running with config run name : {run_name} with config : {cfg}")
        if not os.path.exists(cfg["write_config_directory"]):
            os.makedirs(cfg["write_config_directory"])
        with open(
            os.path.join(cfg["write_config_directory"], f"{run_name}.yaml"), "w"
        ) as f:
            OmegaConf.save(cfg, f)
        if "commit" in cfg and not cfg["commit"]:
            print("Skipping commit")
        else:
            commit(cwd=SRC_DIRECTORY, msg=f"Pre-run commit for : {run_name}")
        # Run the main function
        func(run_name, cfg)
        # Post-run commit
        cfg["finished_running"] = True
        with open(
            os.path.join(cfg["write_config_directory"], f"{run_name}.yaml"), "w"
        ) as f:
            OmegaConf.save(cfg, f)
        if "commit" in cfg and not cfg["commit"]:
            print("Skipping commit")
        else:
            commit(cwd=SRC_DIRECTORY, msg=f"Post-run commit for : {run_name}")

    return wrapper
