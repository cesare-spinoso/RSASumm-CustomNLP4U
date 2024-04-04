import itertools

import pandas as pd
import pytest


@pytest.fixture
def required_columns(dataset_name):
    common_columns = ["document_id", "document", "summary"]
    if dataset_name == "covidet":
        return common_columns + ["emotion"]
    elif dataset_name == "debatepedia":
        return common_columns + ["query", "topic"]
    elif dataset_name == "duc_single":
        return common_columns + ["question", "topic"]
    elif dataset_name == "multioped":
        return common_columns + ["question", "answer"]
    elif dataset_name == "qmsum":
        return common_columns + ["question"]


@pytest.fixture
def dataset_paths(dataset_name):
    ds_paths = {
        "covidet": {
            "all": "/home/mila/c/cesare.spinoso/RSASumm/data/covidet/preprocessed.csv",
            "train": "/home/mila/c/cesare.spinoso/RSASumm/data/covidet/train.csv",
            "val": "/home/mila/c/cesare.spinoso/RSASumm/data/covidet/val.csv",
            "test": "/home/mila/c/cesare.spinoso/RSASumm/data/covidet/test.csv",
        },
        "debatepedia": {
            "all": "/home/mila/c/cesare.spinoso/RSASumm/data/debatepedia/preprocessed.csv",
            "train": "/home/mila/c/cesare.spinoso/RSASumm/data/debatepedia/train.csv",
            "val": "/home/mila/c/cesare.spinoso/RSASumm/data/debatepedia/val.csv",
            "test": "/home/mila/c/cesare.spinoso/RSASumm/data/debatepedia/test.csv",
        },
        "duc_single": {
            "all": "/home/mila/c/cesare.spinoso/RSASumm/data/duc/duc_single/preprocessed.csv",
        },
        "multioped": {
            "all": "/home/mila/c/cesare.spinoso/RSASumm/data/multioped/preprocessed.csv",
            "train": "/home/mila/c/cesare.spinoso/RSASumm/data/multioped/train.csv",
            "val": "/home/mila/c/cesare.spinoso/RSASumm/data/multioped/val.csv",
            "test": "/home/mila/c/cesare.spinoso/RSASumm/data/multioped/test.csv",
        },
        "qmsum": {
            "all": "/home/mila/c/cesare.spinoso/RSASumm/data/qmsum/preprocessed.csv",
            "train": "/home/mila/c/cesare.spinoso/RSASumm/data/qmsum/train.csv",
            "val": "/home/mila/c/cesare.spinoso/RSASumm/data/qmsum/val.csv",
            "test": "/home/mila/c/cesare.spinoso/RSASumm/data/qmsum/test.csv",
        },
    }
    return ds_paths[dataset_name]


def test_data_lengths(dataset_paths):
    assert (len(dataset_paths) == 1 and "all" in dataset_paths) or (
        len(dataset_paths) == 4
        and all(x in dataset_paths for x in ["all", "train", "val", "test"])
    )
    if len(dataset_paths) == 4:
        df_all = pd.read_csv(dataset_paths["all"])
        assert len(df_all) == sum(
            [len(pd.read_csv(dataset_paths[x])) for x in ["train", "val", "test"]]
        )


def test_cell_entries(dataset_paths, required_columns):
    for ds_path in dataset_paths.values():
        df = pd.read_csv(ds_path)
        assert all(df[col].isna().sum() == 0 for col in required_columns)
        assert all(
            df[col].apply(lambda x: x == "").sum() == 0 for col in required_columns
        )
        assert all(
            df[col].isna().sum() == len(df)
            for col in set(df.columns) - set(required_columns)
        )


def test_data_leakage(dataset_paths):
    if len(dataset_paths) == 4:
        dfs = {
            split: pd.read_csv(dataset_paths[split])
            for split in ["train", "val", "test"]
        }
        for split1, split2 in itertools.combinations(dfs.keys(), 2):
            assert (
                set(dfs[split1].itertuples(index=False, name=None))
                & set(dfs[split2].itertuples(index=False, name=None))
                == set()
            )
