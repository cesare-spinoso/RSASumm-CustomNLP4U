import os
from itertools import combinations

import hydra
import numpy as np
import pandas as pd
import rapidfuzz
from tqdm import tqdm
import yaml

from src import SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import read_jsonlines, write_json_file


def filter_summaries(summaries_jsonl, dataset_name, filter_datasets):
    if filter_datasets is None or dataset_name not in filter_datasets:
        return summaries_jsonl
    df_raw = pd.read_csv(filter_datasets[dataset_name])
    filtered_summaries = []
    for summary in summaries_jsonl:
        if summary["source"] in df_raw["document"].values.tolist():
            filtered_summaries.append(summary)
    return filtered_summaries


def get_summaries_from_yaml(yaml_file_path, filter_datasets=None):
    # NOTE: This might still introduce false positives in the
    # case where a source document occurs more than once in the raw
    # dataset because, e.g., it is associated with many latents
    with open(yaml_file_path, "r") as f:
        yaml_content = yaml.safe_load(f)
    summaries_dict = {}
    for model_name, dataset_dict in yaml_content.items():
        summaries_dict[model_name] = {}
        for dataset_name, summaries_jsonl_path in dataset_dict.items():
            summaries_jsonl = read_jsonlines(summaries_jsonl_path)
            summaries_dict[model_name][dataset_name] = pd.DataFrame(
                filter_summaries(summaries_jsonl, dataset_name, filter_datasets)
            )
    return summaries_dict


def compute_mean_diversity(df_summaries, diversity_metric):
    grouping_columns = ["document_id", "source"]
    grouped_preds = [
        table["pred"].values.tolist()
        for _, table in df_summaries.groupby(grouping_columns)
    ]
    pred_combinations = [list(combinations(preds, 2)) for preds in grouped_preds]
    diversity_scores = np.empty(
        [len(pred_combinations), len(max(pred_combinations, key=lambda x: len(x)))]
    )
    diversity_scores[:, :] = np.nan
    for i, pred_pairs in tqdm(enumerate(pred_combinations)):
        for j, (pred1, pred2) in enumerate(pred_pairs):
            diversity_scores[i, j] = diversity_metric(pred1, pred2)
    return np.mean(np.nanmean(diversity_scores, axis=1))


def compute_random_diversity(df_summaries, diversity_metric):
    grouping_columns = ["document_id", "source"]
    sampled_preds = [
        np.random.choice(table["pred"].values.tolist(), 1)
        for _, table in df_summaries.groupby(grouping_columns)
    ]
    pred_combinations = list(combinations(sampled_preds, 2))
    diversity_scores = np.empty(len(pred_combinations))
    for i, (pred1, pred2) in tqdm(enumerate(pred_combinations)):
        diversity_scores[i] = diversity_metric(pred1, pred2)
    return np.mean(diversity_scores)


def compute_metric_for_dataset(df_summaries, metric_name):
    if metric_name == "LCS":
        diversity_metric = rapidfuzz.distance.LCSseq.normalized_distance
    elif metric_name == "Hamming":
        diversity_metric = rapidfuzz.distance.Hamming.normalized_distance
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
    mean_diversity = compute_mean_diversity(df_summaries, diversity_metric)
    random_diversity = compute_random_diversity(df_summaries, diversity_metric)
    return {metric_name: {"mean": mean_diversity, "random": random_diversity}}


def compute_metric(summaries_dict, metric_name):
    assert metric_name in ["LCS", "Hamming"]
    metric_dict = {}
    for model_name, dataset_dict in summaries_dict.items():
        print(f"Computing metric for {model_name}")
        metric_dict[model_name] = {}
        for dataset_name, df_summaries in dataset_dict.items():
            print(f"Computing metric for {dataset_name}")
            metric_dict[model_name][dataset_name] = compute_metric_for_dataset(
                df_summaries, metric_name
            )
    return metric_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        SRC_DIRECTORY,
        "summary_statistics",
        "conf",
        "summary_diversity",
    ),
)
@main_decorator
def main(run_name, cfg):
    summaries_dict = get_summaries_from_yaml(
        cfg["yaml_file_path"], filter_datasets=cfg.get("filter_datasets", None)
    )
    metric_dict = compute_metric(summaries_dict, cfg["diversity_metric"])
    write_json_file(metric_dict, cfg["output_directory"], run_name)


if __name__ == "__main__":
    main()
