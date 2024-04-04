# TODO: The summaries for llama-2 need some post-processing
import json
import os
import random
from itertools import product

import hydra
import numpy as np
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from src import DATASET_NAMES, SRC_DIRECTORY
from src.evaluation.compute_metrics import compute_rouge
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.decorators import main_decorator
from src.utils.helper import get_jsonl_path_from_yaml


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    assert cfg["summarizer_name"] == hydra_job["config_name"]
    assert set(cfg["datasets"].keys()) <= DATASET_NAMES
    if "merged_summaries_dir" in cfg:
        merged_summaries_dir = cfg["merged_summaries_dir"]
    elif "merged_rescores_yaml" in cfg:
        yaml_file_path = cfg["merged_rescores_yaml"]
        with open(yaml_file_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
        merged_summaries_dir = yaml_cfg[cfg["summarizer_name"]]
    else:
        raise ValueError("No merged_summaries_dir or merged_rescores_yaml in config")
    cut_dir = merged_summaries_dir.split("/")[-2:]
    if "_filtered" in cut_dir[1]:
        cut_dir[1] = cut_dir[1].replace("_filtered", "")
    yaml_file_path = os.path.join(
        SRC_DIRECTORY,
        "rescoring",
        "config_instances",
        cut_dir[0],
        f"{cut_dir[1]}.yaml",
    )
    with open(yaml_file_path, "r") as f:
        yaml_cfg = yaml.safe_load(f)
        assert yaml_cfg["summarizer_name"] == cfg["summarizer_name"]


def precompute_metric(
    metric_name,
    dataset_name,
    preprocessed_df,
    merged_pred_df,
):
    grouping_columns = ["source", get_latent_column_name(dataset_name)]
    preprocessed_df = preprocessed_df.rename(columns={"document": "source"})
    preprocessed_df = preprocessed_df[grouping_columns + ["summary"]]
    merged_ref_pred = pd.merge(
        preprocessed_df,
        merged_pred_df,
        on=grouping_columns,
    )
    merged_ref_pred = merged_ref_pred.dropna()
    grouped_df = {k: table for k, table in merged_ref_pred.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in grouped_df.values():
        cross_product = list(
            product(
                group_table["summary"].unique().tolist(),
                group_table["pred"].unique().tolist(),
            )
        )
        references += [r for r, _ in cross_product]
        summaries += [s for _, s in cross_product]
    assert (len_ := len(summaries)) == len(references)
    if metric_name == "rouge":
        print(
            f"Pre-computing rouge scores for {len_} pairs. This might take a while..."
        )
        compute_rouge(
            predictions=summaries,
            references=references,
        )
    else:
        raise ValueError(f"Unknown/unsupported metric computation: {metric_name}")


def get_summ_ref_pairing(
    dataset_name, evaluation, preprocessed_df, merged_pred_df, lambda_=None, alpha=None
):
    # Merge the two dataframes
    grouping_columns = ["source", get_latent_column_name(dataset_name)]
    preprocessed_df = preprocessed_df.rename(columns={"document": "source"})
    preprocessed_df = preprocessed_df[grouping_columns + ["summary"]]
    merged_ref_pred = pd.merge(
        preprocessed_df,
        merged_pred_df,
        on=grouping_columns,
    )
    merged_ref_pred = merged_ref_pred.dropna()
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
        elif evaluation == "weighted_rsa":
            assert alpha is not None and lambda_ is not None
            temp_df = group_table[
                ["pred", "pred_score", "source_avg_rec_score", "latent_avg_rec_score"]
            ].drop_duplicates()
            # lambda_ = 1 is direct, lambda_ = 0 alpha = 1 is source rec, alpha = 0 is latent rec
            temp_df["weighted_rsa_score"] = lambda_ * temp_df["pred_score"] + (
                1 - lambda_
            ) * (
                alpha * temp_df["source_avg_rec_score"]
                + (1 - alpha) * temp_df["latent_avg_rec_score"]
            )
            summaries.append(
                temp_df.sort_values("weighted_rsa_score", axis=0, ascending=False)[
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


def clean_llama_predictions(merged_df):
    print("Removing [/INST] from Llama output")
    merged_df["pred"] = merged_df["pred"].apply(
        lambda x: x if "[/INST]" not in x else x.split("[/INST]")[-1]
    )
    return merged_df


def filter_for_common_denom(dataset_name, merged_df, merged_rescore_yaml):
    latent_column_name = get_latent_column_name(dataset_name)
    common_denom = get_common_denom(dataset_name, merged_rescore_yaml)
    df_filtered = merged_df[
        merged_df[["source", latent_column_name]].apply(
            lambda x: (x["source"], x[latent_column_name]) in common_denom, axis=1
        )
    ]
    return df_filtered


def get_common_denom(dataset_name, merged_rescore_yaml):
    with open(merged_rescore_yaml, "r") as f:
        yaml_contents = yaml.safe_load(f)
    dataframes = [
        pd.read_csv(os.path.join(merged_path, f"{dataset_name}.csv"))
        for merged_path in yaml_contents.values()
    ]
    latent_column_name = get_latent_column_name(dataset_name)
    sources_and_topics = [
        list(zip(df["source"], df[latent_column_name])) for df in dataframes
    ]
    common_denom = set.intersection(*map(set, sources_and_topics))
    return common_denom


def get_system_outputs(merged_df, preprocessed_df, dataset_name, common_denom, n):
    latent_column_name = get_latent_column_name(dataset_name)
    grouping_columns = ["source", latent_column_name]
    preprocessed_df = preprocessed_df.rename(columns={"document": "source"})
    preprocessed_df = preprocessed_df[grouping_columns + ["summary"]]
    merged_df = pd.merge(
        preprocessed_df,
        merged_df,
        on=grouping_columns,
    )
    subset_denoms = list(common_denom)[:n]
    df_subset = merged_df[
        merged_df[["source", latent_column_name]].apply(
            lambda x: (x["source"], x[latent_column_name]) in subset_denoms, axis=1
        )
    ]
    return df_subset[
        [
            "source",
            latent_column_name,
            "summary",
            "pred",
            "pred_score",
            "source_avg_rec_score",
            "latent_avg_rec_score",
        ]
    ].to_json()


def evaluate_summaries(dataset_name, cfg) -> dict:
    # Preprocess input needed to compute metrics
    preprocessed_dataset_path = cfg["datasets"][dataset_name]
    preprocessed_df = pd.read_csv(preprocessed_dataset_path)
    # Read merged dataframe with pred_score, and reconstruction scores
    if "merged_summaries_dir" in cfg:
        merged_summaries_dir = cfg["merged_summaries_dir"]
    else:
        merged_summaries_dir = get_jsonl_path_from_yaml(
            [cfg["summarizer_name"]], cfg["merged_rescores_yaml"]
        )
    merged_df = pd.read_csv(os.path.join(merged_summaries_dir, f"{dataset_name}.csv"))
    if "precompute_only" not in cfg["evaluations"] and cfg.get("ensure_common_denom"):
        # NOTE: This is a patch for now since some of the test summaries are missing
        # This ensures that a row is kept only if it exists in all the other merges
        merged_df = filter_for_common_denom(
            dataset_name,
            merged_df,
            merged_rescore_yaml=os.path.join(
                SRC_DIRECTORY, "rescoring", "merged_rescores", "merged_rescores.yaml"
            ),
        )
    # Clean the merged df e.g. for Llama2 outputs
    if "llama" in cfg["summarizer_name"]:
        merged_df = clean_llama_predictions(merged_df)
    if "avg_source_rec_score" in merged_df.columns:
        merged_df = merged_df.rename(
            columns={
                "avg_source_rec_score": "source_avg_rec_score",
            }
        )
    # Match the merged summaries with the refereces from preprocessed_dataset_path
    evaluation_dict = dict.fromkeys(cfg["evaluations"])
    for evaluation in evaluation_dict:
        print(f"Evaluating for {evaluation}")
        if evaluation in [
            "best",
            "random",
            "direct",
            "source_reconstruction",
            "latent_reconstruction",
        ]:
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
        elif evaluation == "system_output":
            common_denom = get_common_denom(
                dataset_name,
                merged_rescore_yaml=os.path.join(
                    SRC_DIRECTORY,
                    "rescoring",
                    "merged_rescores",
                    "merged_rescores.yaml",
                ),
            )
            evaluation_dict[evaluation] = get_system_outputs(
                merged_df, preprocessed_df, dataset_name, common_denom, n=5
            )
        elif evaluation == "precompute_only":
            # Precompute the metric for all possible ref-summ pairs
            evaluation_dict[evaluation] = None
            precompute_metric(
                metric_name=cfg["metric"],
                dataset_name=dataset_name,
                preprocessed_df=preprocessed_df,
                merged_pred_df=merged_df,
            )
        elif evaluation == "weighted_rsa":
            # Precompute the metric for all possible ref-summ pairs
            evaluation_dict[evaluation] = {}
            precompute_metric(
                metric_name=cfg["metric"],
                dataset_name=dataset_name,
                preprocessed_df=preprocessed_df,
                merged_pred_df=merged_df,
            )
            lambdas = np.arange(0, 1.1, cfg["lambda_interval"])
            alphas = np.arange(0, 1.1, cfg["alpha_interval"])
            cross_product = product(lambdas, alphas)
            for lambda_, alpha in tqdm(cross_product):
                if lambda_ == 1.0 and alpha != 0.0:
                    continue
                print(f"Weighted RSA: lambda={lambda_}, alpha={alpha}")
                summaries, references = get_summ_ref_pairing(
                    dataset_name=dataset_name,
                    evaluation=evaluation,
                    preprocessed_df=preprocessed_df,
                    merged_pred_df=merged_df,
                    lambda_=lambda_,
                    alpha=alpha,
                )
                # Convert to string because json doesn't accept tuples as keys
                evaluation_dict[evaluation][str((lambda_, alpha))] = compute_metric(
                    metric_name=cfg["metric"],
                    summaries=summaries,
                    references=references,
                )
    return evaluation_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "evaluation", "conf"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    if any(elt in cfg["summarizer_name"] for elt in ["generic", "e2e"]):
        dataset_names = cfg["datasets"].keys()
        evaluation_json = {}
        output_path = os.path.join(cfg["output_directory"], f"{run_name}.json")
        for ds_name in dataset_names:
            print(f"Evaluating {ds_name}")
            evaluation_json = {
                ds_name: evaluate_summaries(ds_name, cfg),
                **evaluation_json,
            }
            with open(output_path, "w") as f:
                json.dump(evaluation_json, f, indent=4)
    else:
        raise ValueError(f"Unknown summarizer name: {cfg['summarizer_name']}")


if __name__ == "__main__":
    main()
