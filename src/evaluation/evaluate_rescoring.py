import json
import os
import random
from itertools import product

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src import SRC_DIRECTORY
from src.evaluation.compute_metrics import compute_rouge
from src.merging.cache_merging import convert_emotion_to_question
from src.utils.dataset import get_question_column_name
from src.utils.decorators import main_decorator


def sanity_check_config(cfg):
    pass


def get_summ_ref_pairing(
    dataset_name, merged_dataset, evaluation, lambda_=None, alpha=None
):
    grouping_columns = ["document", get_question_column_name(dataset_name)]
    grouped_df = {k: table for k, table in merged_dataset.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in tqdm(grouped_df.values(), desc="Summ-Ref"):
        references.append(group_table["summary"].unique().tolist())
        # Remove any na/empty preds (occurs with some summarizers)
        group_table = group_table[~group_table["pred"].isna()]
        assert len(group_table) >= 1
        if evaluation == "best":
            summaries.append(group_table["pred"].unique().tolist())
        elif evaluation == "random":
            summaries.append(random.choice(group_table["pred"].unique().tolist()))
        elif evaluation == "direct":
            # Some summaries are the same but have different scores
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
        for i, (summary, reference) in enumerate(
            tqdm(zip(summaries, references), total=len(summaries), desc="Rouge-Comp")
        ):
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


def precompute_metric(
    metric_name,
    merged_dataset,
    dataset_name,
):
    grouping_columns = ["document", get_question_column_name(dataset_name)]
    grouped_df = {k: table for k, table in merged_dataset.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in grouped_df.values():
        # Same fix as in get_summ_ref_pairing
        group_table = group_table[~group_table["pred"].isna()]
        assert len(group_table) >= 1
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


def evaluate_summaries(merged_dataset, dataset_name, cfg):
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
                merged_dataset=merged_dataset,
                evaluation=evaluation,
            )
            evaluation_dict[evaluation] = compute_metric(
                metric_name=cfg["metric"],
                summaries=summaries,
                references=references,
            )
        elif evaluation in ["precompute_only", "weighted_rsa"]:
            # Precompute the metric for all possible ref-summ pairs
            evaluation_dict[evaluation] = {}
            precompute_metric(
                metric_name=cfg["metric"],
                merged_dataset=merged_dataset,
                dataset_name=dataset_name,
            )
            if evaluation == "weighted_rsa":
                lambdas = np.arange(0, 1.1, cfg["lambda_interval"])
                alphas = np.arange(0, 1.1, cfg["alpha_interval"])
                cross_product = product(lambdas, alphas)
                for lambda_, alpha in tqdm(cross_product):
                    if lambda_ == 1.0 and alpha != 0.0:
                        continue
                    print(f"Weighted RSA: lambda={lambda_}, alpha={alpha}")
                    summaries, references = get_summ_ref_pairing(
                        dataset_name=dataset_name,
                        merged_dataset=merged_dataset,
                        evaluation=evaluation,
                        lambda_=lambda_,
                        alpha=alpha,
                    )
                    # Convert to string because json doesn't accept tuples as keys
                    evaluation_dict[evaluation][str((lambda_, alpha))] = compute_metric(
                        metric_name=cfg["metric"],
                        summaries=summaries,
                        references=references,
                    )
        else:
            raise ValueError(f"Unknown evaluation: {evaluation}")
    return evaluation_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        SRC_DIRECTORY, "evaluation", "conf", "evaluate_cache_merged"
    ),
    config_name="config",
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    if "cache_merging_dir" in cfg:
        # Use the cache dir instead
        output_dir_path = os.path.join(cfg["output_directory"], run_name)
        if not os.path.exists(output_dir_path):
            os.makedirs(output_dir_path)
        for summarizer_name in os.listdir(cfg["cache_merging_dir"]):
            evaluation_json = {}
            if "summarizer_names" in cfg and summarizer_name not in cfg["summarizer_names"]:
                continue
            for dataset_filename in os.listdir(
                os.path.join(cfg["cache_merging_dir"], summarizer_name)
            ):
                dataset_name = dataset_filename.replace(".csv", "")
                merged_dataset = pd.read_csv(
                    os.path.join(
                        cfg["cache_merging_dir"], summarizer_name, dataset_filename
                    )
                )
                if dataset_name == "covidet":
                    merged_dataset["question"] = convert_emotion_to_question(
                        merged_dataset
                    )
                output_path = os.path.join(output_dir_path, f"{summarizer_name}.json")
                evaluation_json[dataset_name] = evaluate_summaries(
                    merged_dataset, dataset_name, cfg
                )
                with open(output_path, "w") as f:
                    json.dump(evaluation_json, f, indent=4)
    else:
        raise ValueError(f"Unsupported evaluation: {cfg}")


if __name__ == "__main__":
    main()
