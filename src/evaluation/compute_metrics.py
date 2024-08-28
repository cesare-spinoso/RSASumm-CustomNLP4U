"""
Compute metrics while caching their computation.
"""

import fcntl
import os
import pickle as pkl
from typing import Union
from filelock import FileLock

import evaluate
import torch
import torch.nn.functional as F
import torch.utils.data
from bert_score import BERTScorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src import METRICS_CACHE_DIR, SCRATCH_CACHE_DIR

UNARY_METRICS_FILENAME = "unary_metrics"
BINARY_METRICS_FILENAME = "binary_metrics"
METRICS = {
    "rouge",
    "rouge1",
    "rouge2",
    "rougeL",
    "meteor",
    "bertscore",
    "seahorse-comprehensible",
    "seahorse-repetition",
    "seahorse-grammar",
    "seahorse-attribution",
    "seahorse-mainideas",
    "seahorse-conciseness",
}

map_metric_name_to_question_num = {
    "seahorse-comprehensible": 1,
    "seahorse-repetition": 2,
    "seahorse-grammar": 3,
    "seahorse-attribution": 4,
    "seahorse-mainideas": 5,
    "seahorse-conciseness": 6,
}


def safe_read(filename):
    with open(filename, "rb") as file:
        # Acquire an exclusive lock for reading
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            data = pkl.load(file)
            # Process the data as needed
            return data
        finally:
            # Release the lock
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def safe_write(filename, data):
    with open(filename, "wb") as file:
        # Acquire an exclusive lock for writing
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            pkl.dump(data, file)
            # Data has been written
        finally:
            # Release the lock
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def get_pkl_cache_path(cache_type: str, metric_name: str) -> str:
    if cache_type == "unary":
        cache_path = os.path.join(
            METRICS_CACHE_DIR, f"{metric_name}_{UNARY_METRICS_FILENAME}.pkl"
        )
    elif cache_type == "binary":
        cache_path = os.path.join(
            METRICS_CACHE_DIR, f"{metric_name}_{BINARY_METRICS_FILENAME}.pkl"
        )
    else:
        raise ValueError(f"Unsupported cache type: {cache_type}")
    return cache_path


def check_binary_cache(
    first_input: list[str], second_input: list[str], metric_name: str
):
    assert (
        isinstance(first_input, list)
        and isinstance(second_input, list)
        and len(first_input) == len(second_input)
    )
    binary_cache_path = get_pkl_cache_path(cache_type="binary", metric_name=metric_name)
    if not os.path.exists(binary_cache_path):
        safe_write(binary_cache_path, {})
    cached_metrics = safe_read(binary_cache_path)
    # Also check the older file
    older_cached_metrics = safe_read(os.path.join(METRICS_CACHE_DIR, "binary_metrics.pkl"))
    metric_values = []
    for x1, x2 in zip(first_input, second_input):
        if (x1, x2) in cached_metrics and metric_name in cached_metrics[(x1, x2)]:
            metric_values.append(cached_metrics[(x1, x2)][metric_name])
        elif (x1, x2) in older_cached_metrics and metric_name in older_cached_metrics[(x1, x2)]:
            metric_values.append(older_cached_metrics[(x1, x2)][metric_name])
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
    binary_cache_path = get_pkl_cache_path(cache_type="binary", metric_name=metric_name)
    cached_metrics = safe_read(binary_cache_path)
    for x1, x2, value in zip(first_input, second_input, metric_values):
        if (x1, x2) not in cached_metrics:
            cached_metrics[(x1, x2)] = {}
        cached_metrics[(x1, x2)][metric_name] = value
    safe_write(binary_cache_path, cached_metrics)


def cache_unary_metrics(
    first_input: str,
    metric_name: str,
    metric_values: list,
):
    assert len(first_input) == len(metric_values)
    assert metric_name in METRICS
    unary_cache_path = get_pkl_cache_path(cache_type="unary", metric_name=metric_name)
    cached_metrics = safe_read(unary_cache_path)
    for x1, value in zip(first_input, metric_values):
        if x1 not in cached_metrics:
            cached_metrics[x1] = {}
        cached_metrics[x1][metric_name] = value
    safe_write(unary_cache_path, cached_metrics)


def check_unary_cache(first_input: list[str], metric_name: str) -> list:
    assert isinstance(first_input, list)
    unary_cache_path = get_pkl_cache_path(cache_type="unary", metric_name=metric_name)
    if not os.path.exists(unary_cache_path):
        safe_write(unary_cache_path, {})

    cached_metrics = safe_read(unary_cache_path)
    metric_values = []
    for x1 in first_input:
        if x1 in cached_metrics and metric_name in cached_metrics[x1]:
            metric_values.append(cached_metrics[x1][metric_name])
        else:
            metric_values.append(None)
    return metric_values


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, list]:
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


def compute_meteor(predictions: list[str], references: list[str]) -> list:
    assert len(predictions) == len(references)
    # 1. Check the cache
    meteor_scores = check_binary_cache(predictions, references, "meteor")
    # 2. Compute for elements not in the cache
    # Lazy re-computation if one of the rouge scores but not all of them are cached
    uncached_indices = [i for i in range(len(predictions)) if meteor_scores[i] is None]
    if len(uncached_indices) > 0:
        uncached_predictions = [predictions[i] for i in uncached_indices]
        uncached_references = [references[i] for i in uncached_indices]
        meteor = evaluate.load("meteor")
        new_meteor_scores = [
            meteor.compute(predictions=[pred], references=[ref])["meteor"]
            for pred, ref in zip(uncached_predictions, uncached_references)
        ]
        assert len(new_meteor_scores) == len(uncached_indices)
        # 3. Update the cache
        cache_binary_metrics(
            uncached_predictions, uncached_references, "meteor", new_meteor_scores
        )
        for i, uncached_idx in enumerate(uncached_indices):
            meteor_scores[uncached_idx] = new_meteor_scores[i]
    # 4. Return computation
    return {"meteor": meteor_scores}


def compute_bertscore(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "cuda",
    batch_size: int = 64,
    rescale_with_baseline: bool = True,
) -> list:
    assert len(predictions) == len(references)
    # 1. Check the cache
    bert_scores = check_binary_cache(predictions, references, "bertscore")
    # 2. Compute for elements not in the cache
    # Lazy re-computation if one of the rouge scores but not all of them are cached
    uncached_indices = [i for i in range(len(predictions)) if bert_scores[i] is None]
    if len(uncached_indices) > 0:
        uncached_predictions = [predictions[i] for i in uncached_indices]
        uncached_references = [references[i] for i in uncached_indices]
        bertscorer = BERTScorer(
            model_type=model_type,
            lang=lang,
            device=device,
            batch_size=batch_size,
            rescale_with_baseline=rescale_with_baseline,
        )
        _, _, F1 = bertscorer.score(uncached_predictions, uncached_references)
        new_bert_scores = F1.to("cpu").tolist()
        assert len(new_bert_scores) == len(uncached_indices)
        # 3. Update the cache
        cache_binary_metrics(
            uncached_predictions, uncached_references, "bertscore", new_bert_scores
        )
        for i, uncached_idx in enumerate(uncached_indices):
            bert_scores[uncached_idx] = new_bert_scores[i]
    # 4. Return computation
    return {"bertscore": bert_scores}


def get_max_token_length(tokenizer, text):
    return max(len(tokenizer.encode(t, add_special_tokens=False)) for t in text)


def truncate_text(tokenizer, text, length):
    return [tokenizer.decode(tokenizer.encode(t)[:length]) for t in text]


def compute_seahorse_metric(
    predictions: list[str],
    references: list[str],
    metric_name: str,
    batch_size: int,
) -> list:
    """The searhorse metric is P(yes|templated text) for that metric_name."""
    MAX_LEN = 2048
    # Loading model and tokenizer
    question_number = map_metric_name_to_question_num[metric_name]
    model_name = f"google/seahorse-large-q{question_number}"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=SCRATCH_CACHE_DIR,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=SCRATCH_CACHE_DIR)
    # Template the predictions and inputs
    template = "premise: {premise} hypothesis: {hypothesis}"
    max_summary_length = get_max_token_length(tokenizer=tokenizer, text=predictions)
    references = truncate_text(
        tokenizer=tokenizer, text=references, length=MAX_LEN - max_summary_length
    )
    templated_input = [
        template.format(premise=ref, hypothesis=pred)
        for ref, pred in zip(references, predictions)
    ]
    cached_seahorse = check_unary_cache(templated_input, metric_name)
    assert len(cached_seahorse) == len(templated_input)
    not_cached_input = [
        temped_input
        for cache_content, temped_input in zip(cached_seahorse, templated_input)
        if cache_content is None
    ]
    # Compute the metric by batch
    eval_loader = torch.utils.data.DataLoader(not_cached_input, batch_size=batch_size)
    seahorse_metric = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc=f"Computing {metric_name}"):
            # tokenize the batch
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=MAX_LEN,
            )
            # move the inputs to the device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            N_inputs = inputs["input_ids"].shape[0]
            # make decoder inputs to be <pad>
            decoder_input_ids = torch.full(
                (N_inputs, 1),
                tokenizer.pad_token_id,
                dtype=torch.long,
                device=model.device,
            )
            outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            # retrieve logits for the last token and the scores for 0 and 1
            logits = logits[:, -1, [497, 333]]
            # compute the probabilities
            probs = F.softmax(logits, dim=-1)
            prob_yes = probs[:, 1].tolist()
            # append the metrics
            seahorse_metric.extend(prob_yes)
            # cache the computed metrics
            cache_unary_metrics(batch, metric_name, prob_yes)
    seahorse = [
        elt if elt is not None else seahorse_metric.pop(0) for elt in cached_seahorse
    ]
    assert len(seahorse_metric) == 0
    assert len(seahorse) == len(templated_input)
    return seahorse


def compute_seahorse(
    predictions: list[str],
    references: list[str],
    metric_name: str,
    batch_size: int = 8,
) -> dict[str, list]:
    assert metric_name in [
        metric for metric in METRICS if metric.startswith("seahorse-")
    ]
    # 1. Check the cache
    seahorse_scores = check_binary_cache(predictions, references, metric_name)
    # 2. Compute for elements not in the cache
    # Lazy re-computation if one of the rouge scores but not all of them are cached
    uncached_indices = [i for i, seah in enumerate(seahorse_scores) if seah is None]
    if len(uncached_indices) > 0:
        uncached_predictions = [predictions[i] for i in uncached_indices]
        uncached_references = [references[i] for i in uncached_indices]
        new_seahorse_scores = compute_seahorse_metric(
            predictions=uncached_predictions,
            references=uncached_references,
            metric_name=metric_name,
            batch_size=batch_size,
        )
        assert len(new_seahorse_scores) == len(uncached_indices)
        # 3. Update the cache
        cache_binary_metrics(
            uncached_predictions, uncached_references, metric_name, new_seahorse_scores
        )
        for i, uncached_idx in enumerate(uncached_indices):
            seahorse_scores[uncached_idx] = new_seahorse_scores[i]
    # 4. Return computation
    return {metric_name: seahorse_scores}
