import os

import pandas as pd
import pytest

from src import SRC_DIRECTORY
from src.utils.helper import read_yaml


@pytest.fixture
def raw_and_merged_df(dataset_paths, dataset_name, rec_summarizer_name):
    if "test" not in dataset_paths:
        df_raw = pd.read_csv(dataset_paths["all"])
    else:
        df_raw = pd.read_csv(dataset_paths["test"])
    yaml_file_path = os.path.join(
        SRC_DIRECTORY, "rescoring", "merged_rescores", "merged_rescores.yaml"
    )
    yaml_content = read_yaml(yaml_file_path)
    df_merged = pd.read_csv(
        os.path.join(yaml_content[rec_summarizer_name], f"{dataset_name}.csv")
    )
    return df_raw, df_merged


def test_merged_rescores_docsize(raw_and_merged_df):
    (df_raw, df_merged) = raw_and_merged_df
    assert (x1 := len(set(df_raw["document"].values.tolist()))) == (
        x2 := len(set(df_merged["source"].values.tolist()))
    ), f"Raw unique count is {x1} while merged unique count is {x2}"


def test_merged_rescores_doclatent_size(raw_and_merged_df, dataset_name):
    (df_raw, df_merged) = raw_and_merged_df
    latent_column_name = get_latent_column_name(dataset_name)
    assert len(set(zip(df_raw["document"], df[latent_column_name]))) == len(
        set(zip(df_merged["source"], df_merged[latent_column_name]))
    )
