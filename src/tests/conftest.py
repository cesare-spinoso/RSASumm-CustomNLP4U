import pytest


@pytest.fixture(params=["covidet", "debatepedia", "duc_single", "multioped", "qmsum"])
def dataset_name(request):
    return request.param


@pytest.fixture(
    params=[
        "t5_generic",
        "bart_generic",
        "led_generic",
        "peg_generic",
        "llama2_generic",
    ]
)
def generic_summarizer_name(request):
    return request.param


@pytest.fixture(params=["bart_e2e", "led_e2e", "llama2_e2e"])
def e2e_summarizer_name(request):
    return request.param


@pytest.fixture(
    params=[
        "t5_generic",
        "bart_generic",
        "led_generic",
        "peg_generic",
        "llama2_generic",
        "bart_e2e",
        "led_e2e",
        "llama2_e2e",
    ]
)
def rec_summarizer_name(request):
    return request.param


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