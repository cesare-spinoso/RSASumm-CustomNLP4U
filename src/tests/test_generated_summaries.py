import os
import warnings

import pandas as pd
import pytest

from src import SRC_DIRECTORY
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines


@pytest.fixture
def generic_raw_and_jsonl(generic_summarizer_name, dataset_name):
    # Get the summaries generated in the jsonl file
    yaml_file_path = os.path.join(
        SRC_DIRECTORY,
        "generate_summaries",
        "generated_summaries",
        "generic_summaries.yaml",
    )
    jsonl_path = get_jsonl_path_from_yaml(
        [generic_summarizer_name, dataset_name], yaml_file_path
    )
    jsonlines_data = []
    if isinstance(jsonl_path, list):
        for jp in jsonl_path:
            jsonlines_data += read_jsonlines(jp)
    elif isinstance(jsonl_path, str):
        jsonlines_data = read_jsonlines(jsonl_path)
    else:
        raise ValueError("Unexpected type for jsonl_path")
    # Get the raw input text
    if dataset_name == "duc_single":
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", "duc", "duc_single", "preprocessed.csv"
        )
    else:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "preprocessed.csv"
        )
    raw_data = pd.read_csv(raw_dataset_path)
    return raw_data, jsonlines_data


@pytest.fixture
def e2e_raw_and_jsonl(e2e_summarizer_name, dataset_name):
    # Get the summaries generated in the jsonl file
    yaml_file_path = os.path.join(
        SRC_DIRECTORY,
        "generate_summaries",
        "generated_summaries",
        "e2e_summaries.yaml",
    )
    jsonl_path = get_jsonl_path_from_yaml(
        [e2e_summarizer_name, dataset_name], yaml_file_path
    )
    if jsonl_path is None or isinstance(jsonl_path, list):
        return None, None
    jsonlines_data = read_jsonlines(jsonl_path)
    # Get the raw input text
    if dataset_name == "duc_single":
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", "duc", "duc_single", "preprocessed.csv"
        )
    else:
        raw_dataset_path = os.path.join(
            SRC_DIRECTORY, "..", "data", dataset_name, "test.csv"
        )
    raw_data = pd.read_csv(raw_dataset_path)
    if dataset_name == "covidet":
        # This is not the question exactly, but the question
        # is created deterministically from the emotion
        raw_data["question"] = raw_data["emotion"]
    elif dataset_name == "debatepedia":
        raw_data["question"] = raw_data["query"]
    return raw_data, jsonlines_data


def test_generic_summaries(generic_raw_and_jsonl):
    (raw_data, jsonlines_data) = generic_raw_and_jsonl
    if raw_data is None and jsonlines_data is None:
        assert True == False
    if (x1 := 5 * len(raw_data["document"].unique())) != (x2 := len(jsonlines_data)):
        warnings.warn(
            f"Expected {x1} summaries, got {x2} instead. The difference is {x1 - x2}"
        )


def test_e2e_summaries(e2e_raw_and_jsonl):
    (raw_data, jsonlines_data) = e2e_raw_and_jsonl
    if raw_data is None and jsonlines_data is None:
        assert True == False
    duplicated_data = raw_data[["document", "question"]].duplicated(keep="first")
    raw_data = raw_data[~duplicated_data]
    if (x1 := 5 * len(raw_data)) != (x2 := len(jsonlines_data)):
        warnings.warn(
            f"Expected {x1} summaries, got {x2} instead. The difference is {x1 - x2}"
        )
