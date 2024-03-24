# TODO: The summaries for llama-2 need some post-processing
import json
import os
import random
from itertools import product

import evaluate
import hydra
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import DATASET_NAMES, SRC_DIRECTORY
from src.evaluation.compute_metrics import compute_rouge
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.decorators import main_decorator
from src.utils.helper import (
    get_jsonl_path_from_yaml,
    read_jsonlines,
)


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    assert cfg["summarizer_name"] == hydra_job["config_name"]
    assert set(cfg["datasets"].keys()) <= DATASET_NAMES
    merged_summaries_dir = cfg["merged_summaries_dir"]
    cut_dir = merged_summaries_dir.split("/")[-2:]
    yaml_file_path = os.path.join(
        SRC_DIRECTORY, "rescoring", "config_instances", cut_dir[0], f"{cut_dir[1]}.yaml"
    )
    with open(yaml_file_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)
        assert yaml_cfg["summarizer_name"] == cfg["summarizer_name"]


def get_summ_ref_pairing(dataset_name, evaluation, preprocessed_df, merged_pred_df):
    # Merge the two dataframes
    grouping_columns = ["document_id", "source", get_latent_column_name(dataset_name)]
    preprocessed_df = preprocessed_df.rename(columns={"document": "source"})
    preprocessed_df = preprocessed_df[grouping_columns + ["summary"]]
    merged_ref_pred = pd.merge(
        preprocessed_df,
        merged_pred_df,
        on=grouping_columns,
    )
    grouped_df = {k: table for k, table in merged_ref_pred.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in grouped_df.values():
        references.append(group_table["summary"].unique().tolist())
        if evaluation == "best":
            summaries.append(group_table["pred"].unique().tolist())
        elif evaluation == "random":
            summaries.append(random.choice(group_table["pred"].unique().tolist()))
        elif evaluation == "direct":
            temp_df = group_table[["pred", "pred_score"]].drop_duplicates()
            summaries.append(
                temp_df.sort_values("pred_score", axis=0, ascending=False)[
                    "pred"
                ].values[0]
            )
        elif evaluation == "source_reconstruction":
            temp_df = group_table[["pred", "source_rec_score"]].drop_duplicates()
            summaries.append(
                temp_df.sort_values("source_rec_score", axis=0, ascending=False)[
                    "pred"
                ].values[0]
            )
        elif evaluation == "latent_reconstruction":
            temp_df = group_table[["pred", "latent_rec_score"]].drop_duplicates()
            summaries.append(
                temp_df.sort_values("latent_rec_score", axis=0, ascending=False)[
                    "pred"
                ].values[0]
            )
        else:
            raise ValueError(f"Unknown evaluation: {evaluation}")
    assert len(summaries) == len(references)
    return summaries, references


def compute_metric(metric_name, summaries, references):
    if metric_name == "rouge":
        expanded_indices = []
        expanded_summaries = []
        expanded_references = []
        for i, (summary, reference) in enumerate(zip(summaries, references)):
            summary = summary if isinstance(summary, list) else [summary]
            reference = reference if isinstance(reference, list) else [reference]
            combinations = list(product(summary, reference))
            all_summaries = [s for s, _ in combinations]
            all_references = [r for _, r in combinations]
            expanded_indices.extend([i] * len(combinations))
            expanded_summaries.extend(all_summaries)
            expanded_references.extend(all_references)
        computed_rouge_scores = compute_rouge(
            predictions=expanded_summaries, references=expanded_references
        )
        df = pd.DataFrame({"index": expanded_indices, **computed_rouge_scores})
        return df.groupby("index").max().mean(axis=0).to_dict()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def evaluate_generic_summaries(dataset_name, cfg) -> dict:
    # Preprocess input needed to compute metrics
    preprocessed_dataset_path = cfg["datasets"][dataset_name]
    preprocessed_df = pd.read_csv(preprocessed_dataset_path)
    # Read merged dataframe with pred_score, and reconstruction scores
    merged_df = pd.read_csv(
        os.path.join(cfg["merged_summaries_dir"], f"{dataset_name}.csv")
    )
    # Match the merged summaries with the refereces from preprocessed_dataset_path
    evaluation_dict = dict.fromkeys(cfg["evaluations"])
    for evaluation in evaluation_dict:
        summaries, references = get_summ_ref_pairing(
            dataset_name=dataset_name,
            evaluation=evaluation,
            preprocessed_df=preprocessed_df,
            merged_pred_df=merged_df,
        )
        evaluation_dict[evaluation] = compute_metric(
            metric_name=cfg["metric"],
            summaries=summaries,
            references=references,
        )
    return evaluation_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "evaluation", "conf", "generic_summaries"),
    config_name="t5_generic",
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    if "generic" in cfg["summarizer_name"]:
        dataset_names = cfg["datasets"].keys()
        evaluation_json = {}
        output_path = os.path.join(cfg["output_directory"], f"{run_name}.json")
        for ds_name in dataset_names:
            evaluation_json = {
                ds_name: evaluate_generic_summaries(ds_name, cfg),
                **evaluation_json,
            }
            with open(output_path, "w") as f:
                json.dump(evaluation_json, f, indent=4)
    else:
        raise ValueError(f"Unknown summarizer name: {cfg['summarizer_name']}")


if __name__ == "__main__":
    main()
