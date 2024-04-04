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
