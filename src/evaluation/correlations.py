from itertools import product
import os
import click
import pandas as pd
from copy import deepcopy
import numpy as np

from src import SCRATCH_DIR, SRC_DIRECTORY
from src.evaluation.compute_metrics import (
    compute_bertscore,
    compute_meteor,
    compute_rouge,
)
from src.evaluation.evaluate_rescoring_qfs import get_summ_ref_pairing_likelihood_scores
from src.evaluation.tables import DATASETS
from src.utils.helper import read_json_file, read_yaml
from tabulate import tabulate
from ast import literal_eval


def get_and_write_evals_and_scores(
    merged_yaml_path: str,
    eval_yaml_path: str,
    metrics: list[str],
    evaluations: list[str],
) -> dict:
    """Dict will be 4 levels deep: model_name -> decoding_type -> dataset
    and will point to a table with the merged dataframe, the (added) likelihood scores,
    and the calculated metrics which should hopefully be cached."""
    # Compute the metrics for each row in the merged dataframe
    merged_yaml = read_yaml(merged_yaml_path)
    eval_yaml = read_yaml(eval_yaml_path)
    evals_and_scores_tables = {}
    for model_name, decoding_types in merged_yaml.items():
        evals_and_scores_tables[model_name] = {}
        for decoding_type, merged_path in decoding_types.items():
            if merged_path is None:
                continue
            evals_and_scores_tables[model_name][decoding_type] = {}
            for dataset in DATASETS:
                print("="*60)
                print(f"Computing {model_name=} {decoding_type=} {dataset=}")
                dir_path = os.path.join(
                    SCRATCH_DIR,
                    "rsasumm",
                    "evaluation",
                    "correlations",
                    model_name,
                    decoding_type,
                )
                file_name = f"{dataset}.csv"
                if os.path.exists(os.path.join(dir_path, file_name)):
                    evals_and_scores = pd.read_csv(os.path.join(dir_path, file_name))
                    if evals_and_scores.isna().sum().sum() == 0:
                        print(f"Skipping {model_name=} {decoding_type=} {dataset=}")
                        continue
                os.makedirs(dir_path, exist_ok=True)
                evals_and_scores_tables[model_name][decoding_type][dataset] = {}
                evals_and_scores_table = pd.DataFrame(
                    columns=pd.MultiIndex.from_product(
                        [metrics, ["metric_value", *evaluations]]
                    )
                )
                merged = pd.read_csv(os.path.join(merged_path, f"{dataset}.csv"))
                for metric in metrics:
                    if "rouge" in metric:
                        evals_and_scores_table[(metric, "metric_value")] = (
                            compute_rouge(
                                predictions=merged["generated_summary"].tolist(),
                                references=merged["reference_summary"].tolist(),
                            )[metric]
                        )
                    elif metric == "meteor":
                        evals_and_scores_table[(metric, "metric_value")] = compute_meteor(
                            predictions=merged["generated_summary"].tolist(),
                            references=merged["reference_summary"].tolist(),
                        )[metric]
                    elif metric == "bertscore":
                        evals_and_scores_table[(metric, "metric_value")] = compute_bertscore(
                            predictions=merged["generated_summary"].tolist(),
                            references=merged["reference_summary"].tolist(),
                        )[metric]
                    for evaluation in evaluations:
                        if "rsa" in evaluation:
                            lambda_, alpha = get_max_lambda_alpha(
                                eval_yaml=eval_yaml,
                                metric=metric,
                                model_name=model_name,
                                decoding_type=decoding_type,
                                dataset=dataset,
                                evaluation=evaluation,
                            )
                            (
                                group_ids,
                                evals_and_scores_table[(metric, evaluation)],
                            ) = get_summ_ref_pairing_likelihood_scores(
                                df_merged=merged,
                                evaluation=evaluation,
                                lambda_=lambda_,
                                alpha=alpha,
                            )
                        else:
                            (
                                group_ids,
                                evals_and_scores_table[(metric, evaluation)],
                            ) = get_summ_ref_pairing_likelihood_scores(
                                df_merged=merged, evaluation=evaluation
                            )
                    evals_and_scores_table[("", "group_ids")] = group_ids
                    evals_and_scores_tables[model_name][decoding_type][
                        dataset
                    ] = evals_and_scores_table
                    evals_and_scores_table.to_csv(os.path.join(dir_path, file_name))
    return evals_and_scores_tables


def get_max_lambda_alpha(
    eval_yaml: dict,
    metric: str,
    model_name: str,
    decoding_type: str,
    dataset: str,
    evaluation: str,
) -> float:
    eval_dict = read_json_file(
        os.path.join(eval_yaml[metric][model_name][decoding_type], f"{dataset}.json")
    )
    eval_values = eval_dict[evaluation]
    max_value = -100
    max_lambda, max_alpha = None, None
    for lambda_alpha, metric_values in eval_values.items():
        if metric_values[metric] > max_value:
            max_value = metric_values[metric]
            lambda_alpha = literal_eval(lambda_alpha)
            if isinstance(lambda_alpha, tuple):
                max_lambda, max_alpha = lambda_alpha
            else:
                max_lambda = lambda_alpha
    return max_lambda, max_alpha


@click.command()
@click.option(
    "--merged_yaml_path",
    help="Merged dataset",
    type=str,
    required=True,
    default=SRC_DIRECTORY
    / "rescoring"
    / "merged_rescores"
    / "merged_rescores_split.yaml",
)
@click.option(
    "--eval_yaml_path",
    help="Dataset name",
    type=str,
    required=True,
    default=SRC_DIRECTORY / "evaluation" / "evaluations" / "llama_bart_qfs.yaml",
)
@click.option(
    "--correlation_type",
    help="Correlation type",
    required=True,
    type=click.Choice(
        [
            "likelihood_scores_and_metrics",
        ]
    ),
    default="likelihood_scores_and_metrics",
)
def main(merged_yaml_path: str, eval_yaml_path: str, correlation_type: str) -> None:
    metric_order = ["rouge1", "rouge2", "rougeL", "meteor", "bertscore"]
    evaluations = [
        "random_candidate",
        "S0-only",
        "R1-only-answer",
        "R1-only-source",
        "weighted_rsa_answer_scaled",
        "weighted_rsa_source_scaled",
        "weighted_rsa_hybrid_answer_source_scaled",
    ]
    evals_and_scores = get_and_write_evals_and_scores(
        merged_yaml_path=merged_yaml_path,
        eval_yaml_path=eval_yaml_path,
        metrics=metric_order,
        evaluations=evaluations,
    )
    if correlation_type == "likelihood_scores_and_metrics":
        print(evals_and_scores)
    else:
        raise ValueError(f"Table type {correlation_type} not recognized.")


if __name__ == "__main__":
    main()
