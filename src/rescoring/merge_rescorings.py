from collections import OrderedDict
import os

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import DATASET_NAMES, SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    hydra_runtime = HydraConfig.get().runtime
    assert cfg["summarizer_name"] == hydra_job["config_name"]
    assert any(
        "merge_rescores" in dict_elt["path"]
        for dict_elt in hydra_runtime["config_sources"]
    )
    assert set(cfg["datasets"]) <= DATASET_NAMES
    assert all(
        any(
            k.startswith(s)
            for s in [
                "generic_summaries",
                "source_reconstruction",
                "latent_reconstruction",
            ]
        )
        for k in cfg["yaml_files"]
    )
    assert all(
        "merge_rescores" in cfg[k]
        for k in ["write_config_directory", "output_directory"]
    )


def rename_columns(df, name):
    if name == "source_reconstruction":
        df = df.rename(
            columns={
                "reconstruction_score": "source_rec_score",
                "avg_reconstruction_score": "source_avg_rec_score",
            }
        )
    elif name == "latent_reconstruction":
        df = df.rename(
            columns={
                "reconstruction_score": "latent_rec_score",
                "avg_reconstruction_score": "latent_avg_rec_score",
            }
        )
    return df


def merge_rescored_summaries(dataset_name, cfg):
    # Preprocess input needed to compute metrics
    ordered_keys = [cfg["summarizer_name"], dataset_name]
    # Read jsonlines data
    dfs = OrderedDict()
    for name, yaml_path in cfg["yaml_files"].items():
        jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
        jsonlines_data = read_jsonlines(jsonlines_path)
        df = pd.DataFrame(jsonlines_data)
        # NOTE: Some  predictions are "", we still use them for the merge
        # Pandas conveerts these to NaN, undo this operation
        df["pred"] = df["pred"].fillna("")
        if cfg["rename_columns"]:
            df = rename_columns(df, name)
        dfs[name] = df
    # Merge the three dataframes
    merged_df = None
    for df in dfs.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=cfg["columns_to_merge_on"])
    return merged_df


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf", "merge_rescores"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    for dataset_name in cfg["datasets"]:
        df = merge_rescored_summaries(dataset_name=dataset_name, cfg=cfg)
        os.makedirs(os.path.join(cfg["output_directory"], run_name), exist_ok=True)
        df.to_csv(
            os.path.join(cfg["output_directory"], run_name, f"{dataset_name}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
