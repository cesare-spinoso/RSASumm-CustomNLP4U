import os

import hydra
import pandas as pd

from src import SRC_DIRECTORY, DATASET_NAMES
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.decorators import main_decorator


def sanity_check_config(cfg):
    assert set(cfg["datasets"].keys()) <= DATASET_NAMES


def filter_to_test(df_excess, df_test, dataset_name):
    print(f"Filtering {dataset_name} dataset.")
    latent_column_name = get_latent_column_name(dataset_name)
    df_filtered = df_excess[
        df_excess["source"].isin(df_test["document"])
        & df_excess[latent_column_name].isin(df_test[latent_column_name])
    ]
    print(f"Started with {len(df_excess)} rows, ended with {len(df_filtered)} rows.")
    print(f"Filtered out {len(df_excess) - len(df_filtered)} rows.")
    return df_filtered


def filter_dir(merged_dir, cfg):
    filtered_directory_name = f"{merged_dir}_filtered"
    os.makedirs(filtered_directory_name, exist_ok=True)
    for file_name in os.listdir(merged_dir):
        dataset_name = file_name.split(".")[0]
        assert dataset_name in cfg["datasets"]
        df_excess = pd.read_csv(os.path.join(merged_dir, file_name))
        df_test = pd.read_csv(cfg["datasets"][dataset_name])
        df_filtered = filter_to_test(df_excess, df_test, dataset_name)
        df_filtered.to_csv(
            os.path.join(filtered_directory_name, file_name), index=False
        )


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf", "filter"),
    config_name="config",
)
@main_decorator
def main(_, cfg):
    sanity_check_config(cfg)
    for summarizer, merged_dir in cfg["merged_rescores"].items():
        print(f"Filtering {summarizer} rescores.")
        filter_dir(merged_dir, cfg)


if __name__ == "__main__":
    main()
