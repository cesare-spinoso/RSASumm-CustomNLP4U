import os
import warnings

import pandas as pd
import pytest

from src import SRC_DIRECTORY
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines


@pytest.fixture
def summarizer_type(rec_summarizer_name):
    if "generic" in rec_summarizer_name:
        return "generic"
    elif "e2e" in rec_summarizer_name:
        return "e2e"
    else:
        raise ValueError(f"Unsupported summarizer name {rec_summarizer_name}")


@pytest.fixture
def source_raw_and_rec_jsonl(rec_summarizer_name, dataset_name):
    # Get source reconstruction jsonlines
    yaml_file_path = os.path.join(
        SRC_DIRECTORY,
        "rescoring",
        "rescored_summaries",
        "source_reconstruction.yaml",
    )
    jsonl_path = get_jsonl_path_from_yaml(
        [rec_summarizer_name, dataset_name], yaml_file_path
    )
    if jsonl_path is None:
        assert True == False
    jsonlines_data = read_jsonlines(jsonl_path)
    # Get the raw input text
    if dataset_name == "duc_single":
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", "duc", "duc_single", "preprocessed.csv"
        )
    elif "generic" in rec_summarizer_name:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "preprocessed.csv"
        )
    elif "e2e" in rec_summarizer_name:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "test.csv"
        )
    raw_data = pd.read_csv(raw_dataset_path)
    raw_data["latent"] = raw_data[get_latent_column_name(dataset_name)]
    return raw_data, jsonlines_data


@pytest.fixture
def latent_raw_and_rec_jsonl(rec_summarizer_name, dataset_name):
    # Get source reconstruction jsonlines
    yaml_file_path = os.path.join(
        SRC_DIRECTORY,
        "rescoring",
        "rescored_summaries",
        "latent_reconstruction.yaml",
    )
    jsonl_path = get_jsonl_path_from_yaml(
        [rec_summarizer_name, dataset_name], yaml_file_path
    )
    if jsonl_path is None:
        assert True == False
    jsonlines_data = read_jsonlines(jsonl_path)
    # Get the raw input text
    if dataset_name == "duc_single":
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", "duc", "duc_single", "preprocessed.csv"
        )
    elif "generic" in rec_summarizer_name:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "preprocessed.csv"
        )
    elif "e2e" in rec_summarizer_name:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "test.csv"
        )
    raw_data = pd.read_csv(raw_dataset_path)
    raw_data["latent"] = raw_data[get_latent_column_name(dataset_name)]
    return raw_data, jsonlines_data


def test_source_reconstruction(source_raw_and_rec_jsonl, summarizer_type):
    (raw_data, reconstructed_jsonl) = source_raw_and_rec_jsonl
    if summarizer_type == "generic":
        unique_doc_length = len(raw_data["document_id"].unique())
        if (x1 := 5 * unique_doc_length) != (x2 := len(reconstructed_jsonl)):
            warnings.warn(
                f"Expected {x1} reconstructions, got {x2} instead. The difference is {x1 - x2}"
            )
    if summarizer_type == "e2e":
        duplicated_data = raw_data[["document", "latent"]].duplicated(keep="first")
        raw_data = raw_data[~duplicated_data]
        if (x1 := 5 * len(raw_data)) != (x2 := len(reconstructed_jsonl)):
            warnings.warn(
                f"Expected {x1} reconstructions, got {x2} instead. The difference is {x1 - x2}"
            )


def test_latent_reconstruction(latent_raw_and_rec_jsonl):
    (raw_data, reconstructed_jsonl) = latent_raw_and_rec_jsonl
    duplicated_data = raw_data[["document", "latent"]].duplicated(keep="first")
    raw_data = raw_data[~duplicated_data]
    if (x1 := 5 * len(raw_data)) != (x2 := len(reconstructed_jsonl)):
        warnings.warn(
            f"Expected {x1} reconstructions, got {x2} instead. The difference is {x1 - x2}"
        )
