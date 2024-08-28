import json
import os
import random
from itertools import product

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm
from math import ceil, log10

from src import SRC_DIRECTORY
from src.evaluation.compute_metrics import (
    METRICS,
    compute_bertscore,
    compute_meteor,
    compute_rouge,
    compute_seahorse,
)
from src.utils.decorators import main_decorator
from src.utils.helper import read_yaml


def get_summ_ref_pairing_likelihood_scores(
    df_merged, evaluation, lambda_=None, alpha=None
):
    grouping_columns = ["source", "question"]
    grouped_df = {k: table for k, table in df_merged.groupby(grouping_columns)}
    likelihood_scores = []
    group_ids = []
    for i, group_table in tqdm(enumerate(grouped_df.values()), desc="Summ-Ref-Pairing-Like-Scores"):
        group_ids += [i] * len(group_table)
        if evaluation == "random_candidate":
            # NOTE: Shuffle the pred scores randomly
            likelihood_scores.extend(group_table.sample(frac=1)["pred_score"].tolist())
        elif evaluation == "S0-only":
            # Some summaries are the same but have different scores
            likelihood_scores.extend(group_table["pred_score"].tolist())
        elif evaluation == "R1-only-answer":
            likelihood_scores.extend(group_table["ans_rec_score"].tolist())
        elif evaluation == "R1-only-source":
            likelihood_scores.extend(group_table["source_rec_score"].tolist())
        elif "rsa" in evaluation:
            assert lambda_ is not None
            assert any(
                rsa_type in evaluation for rsa_type in ["answer", "source", "hydrid"]
            )
            temp_df = group_table.copy(deep=True)
            temp_df["S0_score"] = temp_df["pred_score"]
            if "hybrid" in evaluation:
                if "scaled" not in evaluation:
                    temp_df["R1_score"] = (1 - alpha) * temp_df[
                        "ans_rec_score"
                    ] + alpha * temp_df["source_rec_score"]
                else:
                    order_of_ans_rec = ceil(log10(abs(temp_df["ans_rec_score"].mean())))
                    order_of_source_rec = ceil(
                        log10(abs(temp_df["source_rec_score"].mean()))
                    )
                    new_scale = 10 ** (-1 * (order_of_source_rec - order_of_ans_rec))
                    temp_df["R1_score"] = (1 - (alpha * new_scale)) * temp_df[
                        "ans_rec_score"
                    ] + (alpha * new_scale * temp_df["source_rec_score"])
            elif "answer" in evaluation:
                temp_df["R1_score"] = temp_df["ans_rec_score"]
            else:  # source
                temp_df["R1_score"] = temp_df["source_rec_score"]
            if "scaled" not in evaluation:
                # lambda_ = 1 is R1 only, lambda_ = 0 is S0 only
                temp_df["rsa_score"] = (1 - lambda_) * temp_df[
                    "S0_score"
                ] + lambda_ * temp_df["R1_score"]
            else:
                # Make the range of lambda_ be multipied by 10*int(log(avg ans rec score))
                order_of_S0 = ceil(log10(abs(temp_df["S0_score"].mean())))
                order_of_R1 = ceil(log10(abs(temp_df["R1_score"].mean())))
                difference_to_predictions = (
                    order_of_R1 - order_of_S0
                )  # e.g. 10^3 and 10^-1 so 3 - (-1) = 4
                new_scale = 10 ** (-1 * difference_to_predictions)
                temp_df["rsa_score"] = (1 - (lambda_ * new_scale)) * temp_df[
                    "S0_score"
                ] + (lambda_ * new_scale) * temp_df["R1_score"]
            likelihood_scores.extend(temp_df["rsa_score"].tolist())
        else:
            raise ValueError(f"Unknown evaluation: {evaluation}")
    assert len(likelihood_scores) == len(df_merged)
    return group_ids, likelihood_scores

def get_summ_ref_pairing(
    df_merged, evaluation, lambda_=None, alpha=None, reference_free=False
):
    grouping_columns = ["source", "question"]
    grouped_df = {k: table for k, table in df_merged.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in tqdm(grouped_df.values(), desc="Summ-Ref-Pairing"):
        # if reference free like seahorse, then use source as reference
        if not reference_free:
            references.append(group_table["reference_summary"].unique().tolist())
        else:
            references.append(group_table["source"].unique().tolist())
        assert len(group_table) >= 1
        if evaluation == "best_candidate":
            summaries.append(group_table["generated_summary"].unique().tolist())
        elif evaluation == "random_candidate":
            # NOTE: Removed unique, because this affects probability of choice
            summaries.append(random.choice(group_table["generated_summary"].tolist()))
        elif evaluation == "S0-only":
            # Some summaries are the same but have different scores
            temp_df = group_table[["generated_summary", "pred_score"]].drop_duplicates()
            summaries.append(
                temp_df.sort_values("pred_score", axis=0, ascending=False)[
                    "generated_summary"
                ].values[0]
            )
        elif evaluation == "R1-only-answer":
            temp_df = group_table[
                ["generated_summary", "ans_rec_score"]
            ].drop_duplicates()
            summaries.append(
                temp_df.sort_values("ans_rec_score", axis=0, ascending=False)[
                    "generated_summary"
                ].values[0]
            )
        elif evaluation == "R1-only-source":
            temp_df = group_table[
                ["generated_summary", "source_rec_score"]
            ].drop_duplicates()
            summaries.append(
                temp_df.sort_values("source_rec_score", axis=0, ascending=False)[
                    "generated_summary"
                ].values[0]
            )
        elif "rsa" in evaluation:
            assert lambda_ is not None
            assert any(
                rsa_type in evaluation for rsa_type in ["answer", "source", "hydrid"]
            )
            temp_df = group_table.copy(deep=True)
            temp_df["S0_score"] = temp_df["pred_score"]
            if "hybrid" in evaluation:
                if "scaled" not in evaluation:
                    temp_df["R1_score"] = (1 - alpha) * temp_df[
                        "ans_rec_score"
                    ] + alpha * temp_df["source_rec_score"]
                else:
                    order_of_ans_rec = ceil(log10(abs(temp_df["ans_rec_score"].mean())))
                    order_of_source_rec = ceil(
                        log10(abs(temp_df["source_rec_score"].mean()))
                    )
                    new_scale = 10 ** (-1 * (order_of_source_rec - order_of_ans_rec))
                    temp_df["R1_score"] = (1 - (alpha * new_scale)) * temp_df[
                        "ans_rec_score"
                    ] + (alpha * new_scale * temp_df["source_rec_score"])
            elif "answer" in evaluation:
                temp_df["R1_score"] = temp_df["ans_rec_score"]
            else:  # source
                temp_df["R1_score"] = temp_df["source_rec_score"]
            if "scaled" not in evaluation:
                # lambda_ = 1 is R1 only, lambda_ = 0 is S0 only
                temp_df["rsa_score"] = (1 - lambda_) * temp_df[
                    "S0_score"
                ] + lambda_ * temp_df["R1_score"]
            else:
                # Make the range of lambda_ be multipied by 10*int(log(avg ans rec score))
                order_of_S0 = ceil(log10(abs(temp_df["S0_score"].mean())))
                order_of_R1 = ceil(log10(abs(temp_df["R1_score"].mean())))
                difference_to_predictions = (
                    order_of_R1 - order_of_S0
                )  # e.g. 10^3 and 10^-1 so 3 - (-1) = 4
                new_scale = 10 ** (-1 * difference_to_predictions)
                temp_df["rsa_score"] = (1 - (lambda_ * new_scale)) * temp_df[
                    "S0_score"
                ] + (lambda_ * new_scale) * temp_df["R1_score"]
            summaries.append(
                temp_df.sort_values("rsa_score", axis=0, ascending=False)[
                    "generated_summary"
                ].values[0]
            )
        else:
            raise ValueError(f"Unknown evaluation: {evaluation}")
    assert len(summaries) == len(references)
    return summaries, references


def carry_out_metric_computation(metric_name, expanded_summaries, expanded_references):
    if metric_name == "rouge":
        computed_scores = compute_rouge(
            predictions=expanded_summaries, references=expanded_references
        )
    elif metric_name == "meteor":
        computed_scores = compute_meteor(
            predictions=expanded_summaries, references=expanded_references
        )
    elif metric_name == "bertscore":
        computed_scores = compute_bertscore(
            predictions=expanded_summaries, references=expanded_references
        )
    elif "seahorse" in metric_name:
        computed_scores = compute_seahorse(
            predictions=expanded_summaries,
            references=expanded_references,
            metric_name=metric_name,
        )
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
    return computed_scores


def compute_metric(metric_name, summaries, references):
    if metric_name in METRICS:
        expanded_indices = []
        expanded_summaries = []
        expanded_references = []
        for i, (summary, reference) in enumerate(
            tqdm(
                zip(summaries, references),
                total=len(summaries),
                desc="Summ-Ref-Expansion",
            )
        ):
            summary = summary if isinstance(summary, list) else [summary]
            reference = reference if isinstance(reference, list) else [reference]
            combinations = list(product(summary, reference))
            all_summaries = [s for s, _ in combinations]
            all_references = [r for _, r in combinations]
            expanded_indices.extend([i] * len(combinations))
            expanded_summaries.extend(all_summaries)
            expanded_references.extend(all_references)
        computed_scores = carry_out_metric_computation(
            metric_name=metric_name,
            expanded_summaries=expanded_summaries,
            expanded_references=expanded_references,
        )
        df = pd.DataFrame({"index": expanded_indices, **computed_scores})
        return df.groupby("index").max().mean(axis=0).to_dict()
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def precompute_metric(
    metric_name,
    df_merged,
):
    grouping_columns = ["source", "question"]
    grouped_df = {k: table for k, table in df_merged.groupby(grouping_columns)}
    references = []
    summaries = []
    for group_table in grouped_df.values():
        assert len(group_table) >= 1
        cross_product = list(
            product(
                group_table["reference_summary"].unique().tolist(),
                group_table["generated_summary"].unique().tolist(),
            )
        )
        references += [r for r, _ in cross_product]
        summaries += [s for _, s in cross_product]
    assert (len_ := len(summaries)) == len(references)
    print(f"Pre-computing rouge scores for {len_} pairs. This might take a while...")
    carry_out_metric_computation(
        metric_name=metric_name,
        expanded_summaries=summaries,
        expanded_references=references,
    )


def evaluate_qfs_summaries(df_merged, cfg, output_path):
    evaluation_dict = dict.fromkeys(cfg["evaluations"])
    reference_free = "seahorse" in cfg["metric"]
    for evaluation in evaluation_dict:
        print(f"Evaluating for {evaluation}")
        if evaluation in [
            "best_candidate",
            "random_candidate",
            "S0-only",
            "R1-only-answer",
            "R1-only-source",
        ]:
            generated_summaries, reference_summaries = get_summ_ref_pairing(
                df_merged=df_merged,
                evaluation=evaluation,
                reference_free=reference_free,
            )
            evaluation_dict[evaluation] = compute_metric(
                metric_name=cfg["metric"],
                summaries=generated_summaries,
                references=reference_summaries,
            )
        elif evaluation.startswith("weighted_rsa") and "hybrid" not in evaluation:
            # Precompute the metric for all possible ref-summ pairs
            evaluation_dict[evaluation] = {}
            precompute_metric(
                metric_name=cfg["metric"],
                df_merged=df_merged,
            )
            lambdas = np.arange(
                0 + cfg["lambda_interval"],
                1,
                cfg["lambda_interval"],
            )
            lambdas = np.append(lambdas, np.arange(1, 11, 1))
            lambdas = np.append(lambdas, [25, 50, 75, 100])
            for lambda_ in lambdas:
                print(f"Weighted RSA ({evaluation}): lambda={lambda_}")
                summaries, references = get_summ_ref_pairing(
                    df_merged=df_merged,
                    evaluation=evaluation,
                    lambda_=lambda_,
                    reference_free=reference_free,
                )
                evaluation_dict[evaluation][lambda_] = compute_metric(
                    metric_name=cfg["metric"],
                    summaries=summaries,
                    references=references,
                )
        elif evaluation.startswith("weighted_rsa_hybrid"):
            evaluation_dict[evaluation] = {}
            precompute_metric(
                metric_name=cfg["metric"],
                df_merged=df_merged,
            )
            rationality_values = [1.0, 5.0, 10.0, 50.0, 100.0]
            lambdas = [x for x, _ in product(rationality_values, rationality_values)]
            alphas = [x for _, x in product(rationality_values, rationality_values)]
            for lambda_, alpha in zip(lambdas, alphas):
                print(
                    f"Weighted RSA Hybrid ({evaluation}): lambda={lambda_}, alpha={alpha}"
                )
                summaries, references = get_summ_ref_pairing(
                    df_merged=df_merged,
                    evaluation=evaluation,
                    lambda_=lambda_,
                    alpha=alpha,
                    reference_free=reference_free,
                )
                evaluation_dict[evaluation][str((lambda_, alpha))] = compute_metric(
                    metric_name=cfg["metric"],
                    summaries=summaries,
                    references=references,
                )
        else:
            raise ValueError(f"Unknown evaluation: {evaluation}")
        with open(output_path, "w") as f:
            json.dump(evaluation_dict, f, indent=4)
    return evaluation_dict


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "evaluation", "conf", "qfs_summaries"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    summarizer_names = (
        [cfg["summarizer_name"]]
        if "summarizer_name" in cfg
        else cfg["summarizer_names"]
    )
    for summarizer_name in summarizer_names:
        if "qfs" in summarizer_name:
            output_dir_path = os.path.join(
                cfg["output_directory"], run_name, summarizer_name
            )
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
            evaluation_json = {}
            for dataset_name in cfg["dataset_names"]:
                if "merged_scores_dir" in cfg:
                    df_merged = pd.read_csv(
                        os.path.join(cfg["merged_scores_dir"], f"{dataset_name}.csv")
                    )
                elif "merged_scores_yaml" in cfg:
                    merged_scores_yaml = read_yaml(cfg["merged_scores_yaml"])
                    df_merged = pd.read_csv(
                        os.path.join(
                            merged_scores_yaml[summarizer_name], f"{dataset_name}.csv"
                        )
                    )
                else:
                    raise ValueError("No merged scores directory or yaml provided :'-(")
                output_path = os.path.join(output_dir_path, f"{dataset_name}.json")
                evaluation_json[dataset_name] = evaluate_qfs_summaries(df_merged, cfg, output_path)
                with open(output_path, "w") as f:
                    json.dump(evaluation_json[dataset_name], f, indent=4)
        else:
            raise ValueError(f"Unsupported evaluation: {cfg}")


if __name__ == "__main__":
    main()
