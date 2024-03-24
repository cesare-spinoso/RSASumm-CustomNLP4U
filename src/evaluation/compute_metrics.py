"""
Compute metrics while caching their computation.
"""

import os
import pickle as pkl
from typing import Union

import evaluate

from src import METRICS_CACHE_DIR

UNARY_METRICS_FILENAME = "unary_metrics"
BINARY_METRICS_FILENAME = "binary_metrics"
METRICS = {
    "rouge1",
    "rouge2",
    "rougeL",
}


def check_binary_cache(
    first_input: list[str], second_input: list[str], metric_name: str
):
    assert (
        isinstance(first_input, list)
        and isinstance(second_input, list)
        and len(first_input) == len(second_input)
    )
    binary_cache_path = os.path.join(
        METRICS_CACHE_DIR, f"{BINARY_METRICS_FILENAME}.pkl"
    )
    if not os.path.exists(binary_cache_path):
        with open(binary_cache_path, "wb") as f:
            pkl.dump({}, f)
    with open(binary_cache_path, "rb") as f:
        cached_metrics = pkl.load(f)
    metric_values = []
    for x1, x2 in zip(first_input, second_input):
        if (x1, x2) in cached_metrics and metric_name in cached_metrics[(x1, x2)]:
            metric_values.append(cached_metrics[(x1, x2)][metric_name])
        else:
            metric_values.append(None)
    print(
        f"Cache hit is {sum([v is not None for v in metric_values])}/{len(metric_values):.3f}"
    )
    return metric_values


def cache_binary_metrics(
    first_input: list[str],
    second_input: list[str],
    metric_name: str,
    metric_values: list,
):
    assert len(first_input) == len(second_input) == len(metric_values)
    assert metric_name in METRICS
    binary_cache_path = os.path.join(
        METRICS_CACHE_DIR, f"{BINARY_METRICS_FILENAME}.pkl"
    )
    with open(binary_cache_path, "rb") as f:
        cached_metrics = pkl.load(f)
    for x1, x2, value in zip(first_input, second_input, metric_values):
        if (x1, x2) not in cached_metrics:
            cached_metrics[(x1, x2)] = {}
        cached_metrics[(x1, x2)][metric_name] = value
    with open(binary_cache_path, "wb") as f:
        pkl.dump(cached_metrics, f)


def check_unary_cache(first_input: Union[str, list], metric_name: str):
    pass


def compute_rouge(predictions: list[str], references: list[str]):
    assert len(predictions) == len(references)
    # 1. Check the cache
    rouge1 = check_binary_cache(predictions, references, "rouge1")
    rouge2 = check_binary_cache(predictions, references, "rouge2")
    rougeL = check_binary_cache(predictions, references, "rougeL")
    # 2. Compute for elements not in the cache
    # Lazy re-computation if one of the rouge scores but not all of them are cached
    uncached_indices = [
        i
        for i in range(len(predictions))
        if any(r is None for r in [rouge1[i], rouge2[i], rougeL[i]])
    ]
    if len(uncached_indices) > 0:
        uncached_predictions = [predictions[i] for i in uncached_indices]
        uncached_references = [references[i] for i in uncached_indices]
        rouge = evaluate.load("rouge")
        computed_rouge_scores = rouge.compute(
            predictions=uncached_predictions,
            references=uncached_references,
            use_aggregator=False,
        )
        new_rouge1 = computed_rouge_scores["rouge1"]
        new_rouge2 = computed_rouge_scores["rouge2"]
        new_rougeL = computed_rouge_scores["rougeL"]
        assert (
            len(new_rouge1)
            == len(new_rouge2)
            == len(new_rougeL)
            == len(uncached_indices)
        )
        # 3. Update the cache
        cache_binary_metrics(
            uncached_predictions, uncached_references, "rouge1", new_rouge1
        )
        cache_binary_metrics(
            uncached_predictions, uncached_references, "rouge2", new_rouge2
        )
        cache_binary_metrics(
            uncached_predictions, uncached_references, "rougeL", new_rougeL
        )
        for i, uncached_idx in enumerate(uncached_indices):
            rouge1[uncached_idx] = new_rouge1[i]
            rouge2[uncached_idx] = new_rouge2[i]
            rougeL[uncached_idx] = new_rougeL[i]
    # 4. Return computation
    return {"rouge1": rouge1, "rouge2": rouge2, "rougeL": rougeL}
