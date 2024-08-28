from itertools import product
import os
import click
import pandas as pd
from copy import deepcopy
import numpy as np
import plotly.express as px

from src import SCRATCH_DIR, SRC_DIRECTORY
from src.utils.helper import read_json_file, read_yaml
from tabulate import tabulate
from scipy import stats

DATASETS = ["multioped", "qmsum", "squality"]
DATASET_MAPPING = {
    "multioped": "\multioped",
    "qmsum": "\qmsum",
    "squality": "\squality",
}

S1_NAME_MAPPING = {
    "best_candidate": "$\pragoracle$",
    "random_candidate": "$\pragrandom$",
    "S0-only": "$\pragans, \lambda=0$",
    "R1-only-answer": "$\pragans,\lambda=1$",
    "R1-only-source": "$\pragsource,\lambda=1$",
    "weighted_rsa_answer_scaled": "$\pragans,\lambda^*$",
    "weighted_rsa_source_scaled": "$\pragsource,\lambda^*$",
    "weighted_rsa_hybrid_answer_source_scaled": "$\pragas,\lambda^*,\\alpha^*$",
}

METRIC_NAME_MAPPING = {
    "rouge1": "R-1",
    "rouge2": "R-2",
    "rougeL": "R-L",
    "meteor": "METEOR",
    "bertscore": "BERTScore",
    "seahorse-comprehensible": "Comp.",
    "seahorse-repetition": "Rep.",
    "seahorse-grammar": "Gram.",
    "seahorse-attribution": "Attr.",
    "seahorse-mainideas": "Main",
    "seahorse-conciseness": "Conc.",
}

ROUGE_NAME_MAPPING = {
    "R-1": "ROUGE-1",
    "R-2": "ROUGE-2",
    "R-L": "ROUGE-L",
}

MODEL_NAME_MAPPING = {
    "llama3_qfs": "Llama 3",
    "bart_finetuned_qfs": "BART",
}

DECODING_TYPE_MAPPING = {
    "standard_beam": "Beam search decoding (Beam size of 5)",
    "temp_2": "Standard sampling (Temperature of 2)",
    "nucleus_0.95": "Nucleus sampling ($p = 0.95$, Temperature of 1.0)",
    "topk_640": "Top-k sampling ($k = 640$, Temperature of 1.0)",
    "nucleus_0.95_temp_1.2": "Nucleus sampling ($p = 0.95$, Temperature of 1.2)",
    "nucleus_0.95_temp_1.5": "Nucleus sampling ($p = 0.95$, Temperature of 1.5)",
}


def get_evals(eval_yaml_path: str) -> dict:
    """Dict will be 4 levels deep: metric -> model_name -> decoding_type -> dataset"""
    eval_yaml = read_yaml(eval_yaml_path)
    evals = {}
    for metric, model_evals in eval_yaml.items():
        evals[metric] = {}
        for model_name, decoding_types in model_evals.items():
            evals[metric][model_name] = {}
            for decoding_type, eval_dir_path in decoding_types.items():
                evals[metric][model_name][decoding_type] = {}
                if eval_dir_path is None:
                    continue
                for dataset in DATASETS:
                    eval_json_path = os.path.join(eval_dir_path, f"{dataset}.json")
                    if not os.path.exists(eval_json_path):
                        continue
                    model_eval = read_json_file(eval_json_path)
                    evals[metric][model_name][decoding_type][dataset] = model_eval
    return evals


def get_rsa_max_value(s1_eval: dict, metric: str) -> float:
    max_value = 0
    for _, metric_values in s1_eval.items():
        if metric_values[metric] > max_value:
            max_value = metric_values[metric]
    return max_value


def create_table_all_s1(metric: str, eval: dict) -> pd.DataFrame:
    assert (
        eval is not None
    ), "This is probably because the json in the yaml file is incomplete."
    metrics = [metric]
    table_values = []
    table_indices = []
    for s1_type, s1_eval in eval.items():
        values_to_add = []
        table_indices.append(S1_NAME_MAPPING[s1_type])
        for metric in metrics:
            if "rsa" not in s1_type:
                values_to_add.append(s1_eval[metric])
            else:
                values_to_add.append(get_rsa_max_value(s1_eval=s1_eval, metric=metric))
        table_values.append(values_to_add)
    metrics = [METRIC_NAME_MAPPING[metric] for metric in metrics]
    table = pd.DataFrame(table_values, columns=metrics, index=table_indices)
    return table


def create_tables_all_s1(evals: dict) -> dict:
    tables_all_s1 = deepcopy(evals)
    for metric, model_evals in evals.items():
        for model_name, decoding_types in model_evals.items():
            for decoding_type, datasets_eval in decoding_types.items():
                for dataset, eval in datasets_eval.items():
                    table = create_table_all_s1(metric=metric, eval=eval)
                    tables_all_s1[metric][model_name][decoding_type][dataset] = table
    return tables_all_s1


def merge_tables_all_s1_by_decoding_type(tables_all_s1: dict) -> None:
    # Merge by decoding type
    # For each dataset and decoding type, place all the models together
    merged_tables = {}
    for dataset in DATASETS:
        merged_tables[dataset] = {}
        for decoding_type in DECODING_TYPE_MAPPING.keys():
            column_multi_index = pd.MultiIndex.from_product(
                [list(MODEL_NAME_MAPPING.values()), list(METRIC_NAME_MAPPING.values())]
            )
            table = pd.DataFrame(
                np.full((len(S1_NAME_MAPPING), len(column_multi_index)), np.nan),
                columns=column_multi_index,
                index=S1_NAME_MAPPING.values(),
            )
            for model, mapped_model_name in MODEL_NAME_MAPPING.items():
                for metric, mapped_metric_name in METRIC_NAME_MAPPING.items():
                    if dataset not in tables_all_s1[metric][model][decoding_type]:
                        continue
                    table.loc[:, (mapped_model_name, mapped_metric_name)] = (
                        tables_all_s1[metric][model][decoding_type][dataset].loc[
                            :, mapped_metric_name
                        ]
                    )
            merged_tables[dataset][decoding_type] = table
    return merged_tables


def get_metrics_str_for_caption(
    metric_order: list[str], dataset: str, decoding_type: str
) -> str:
    metrics_in_caption = []
    assert set(metric_order) <= {"R-1", "R-2", "R-L", "METEOR", "BERTScore"} or set(
        metric_order
    ) <= {"Comp.", "Rep.", "Gram.", "Conc."}
    if any(r in metric_order for r in ["R-1", "R-2", "R-L"]):
        metrics_in_caption.append("ROUGE")
    if "METEOR" in metric_order:
        metrics_in_caption.append("METEOR")
    if "BERTScore" in metric_order:
        metrics_in_caption.append("BERTScore")
    if "Comp." in metric_order:
        metrics_in_caption.append("Comprehensible (Comp.)")
    if "Rep." in metric_order:
        metrics_in_caption.append("Repetition (Rep.)")
    if "Gram." in metric_order:
        metrics_in_caption.append("Grammar (Gram.)")
    if "Conc." in metric_order:
        metrics_in_caption.append("Conciseness (Conc.)")
    metrics_in_caption.insert(-1, "and")
    metrics_str_caption = ", ".join(metrics_in_caption)
    if any(r in metric_order for r in ["R-1", "R-2", "R-L", "METEOR", "BERTScore"]):
        return (
            f"Performance of the different pragmatic summarizers on the {DATASET_MAPPING[dataset]} dataset measured using the {metrics_str_caption} metrics. "
            + f"The summaries generated by the literal summarizer used {DECODING_TYPE_MAPPING[decoding_type].lower()}."
        )
    else:
        return (
            f"Text quality of the summaries select by different pragmatic summarizers on the {DATASET_MAPPING[dataset]} dataset. "
            + f"The summaries generated by the literal summarizer used {DECODING_TYPE_MAPPING[decoding_type].lower()}. "
            + f"Text quality is measure via the SEAHORSE metrics {metrics_str_caption}."
        )


def reformat_tables_all_s1(
    tables_all_s1: dict,
    model_order: list[str],
    metric_order: list[str],
    s1_order: list[str],
) -> dict:
    reformated_tables_all_s1 = {}
    for dataset in DATASETS:
        reformated_tables_all_s1[dataset] = {}
        for decoding_type in DECODING_TYPE_MAPPING.keys():
            ordered_metrics = list(product(model_order, metric_order))
            original_table = tables_all_s1[dataset][decoding_type]
            reformated_tables_all_s1[dataset][decoding_type] = original_table.loc[
                s1_order, ordered_metrics
            ]
            print(f"Table for {dataset} and {decoding_type}")
            print(reformated_tables_all_s1[dataset][decoding_type])
    return reformated_tables_all_s1


def mark_max(table: pd.DataFrame, ignore: list[str]) -> pd.DataFrame:
    assert isinstance(ignore, list)
    temp_table = table.copy(deep=True)
    table = table.map(lambda x: f"{x:.4f}")
    temp_table.loc[ignore, :] = -1000
    for col in temp_table.columns:
        temp_table[col] = temp_table[col].apply(
            lambda x: (
                f"\\textbf{{{x:.4f}}}" if x == temp_table[col].max() else f"{x:.4f}"
            )
        )
    temp_table.loc[ignore, :] = table.loc[ignore, :]
    return temp_table


def convert_tables_all_s1_to_latex(
    tables_all_s1: dict, model_order: list[str], metric_order: list[str]
) -> dict:
    latex_tables = deepcopy(tables_all_s1)
    for dataset, decoding_type_tables in tables_all_s1.items():
        for decoding_type, table in decoding_type_tables.items():
            table = mark_max(table, ignore=["$\pragoracle$"])
            tabulate_rows = tabulate(
                table,
                tablefmt="latex_raw",
                headers="keys",
                showindex=True,
                # floatfmt=".3f", # NOTE: Assume that mark_max already did .3f
            ).split("\n")
            assert len(tabulate_rows) == 14
            full_length = table.shape[1]
            half_length = int(full_length / 2)
            # Replacements
            tabulate_rows[0] = tabulate_rows[0].replace(
                "l" * (full_length + 1), "l" + "c" * full_length
            )
            tabulate_rows[1] = "\\toprule"
            tabulate_rows[3] = ""
            tabulate_rows[-2] = "\\bottomrule"
            tabulate_rows[2] = (
                f"& \multicolumn{{{half_length}}}{{c}}{{{model_order[0]}}} & \multicolumn{{{half_length}}}{{c}}{{{model_order[1]}}} \\\\"
            )
            cmidrule_line = "\cmidrule(r){{{}-{}}}\cmidrule(r){{{}-{}}}".format(
                2, 1 + half_length, 2 + half_length, 1 + 2 * half_length
            )
            tabulate_rows[4] = tabulate_rows[4] + cmidrule_line
            tabulate_rows[7] = tabulate_rows[7] + cmidrule_line
            tabulate_rows[10] = tabulate_rows[10] + cmidrule_line
            # Insertions within the table
            tabulate_rows.insert(
                3,
                " & "
                + " & ".join([metric_name for _, metric_name in table.columns])
                + " \\\\",
            )
            tabulate_rows.insert(4, cmidrule_line)
            # Additions outside table
            tabulate_rows.insert(0, "\\resizebox{\linewidth}{!}{")
            tabulate_rows.insert(0, "\centering")
            tabulate_rows.insert(0, "\\begin{table}[H]")
            tabulate_rows.append("}")
            metrics_str_caption = get_metrics_str_for_caption(
                metric_order=metric_order, dataset=dataset, decoding_type=decoding_type
            )
            tabulate_rows.append("\caption{" + metrics_str_caption + "}")
            tabulate_rows.append("\label{" + f"tab:{dataset}_{decoding_type}" + "}")
            tabulate_rows.append("\end{table}")
            latex_tables[dataset][decoding_type] = "\n".join(tabulate_rows)
    return latex_tables


def write_all_s1_latex_tables(
    latex_tables: dict, file_name: str, output_dir: str, skip: list[str]
) -> None:
    assert isinstance(skip, list)
    output_path = os.path.join(output_dir, f"{file_name}.tex")
    with open(output_path, "w") as f:
        f.write("")
    for dataset, decoding_type_tables in latex_tables.items():
        for decoding_type, table in decoding_type_tables.items():
            if dataset in skip or decoding_type in skip:
                continue
            with open(output_path, "a") as f:
                # f.write(f"% {dataset} and {decoding_type}\n")
                f.write("\n\n\n")
                f.write(table)
                f.write("\n\n\n")


def write_s1_top_score_count_latex_tables(
    latex_tables: dict, file_name: str, output_dir: str
) -> None:
    output_path = os.path.join(output_dir, f"{file_name}.tex")
    with open(output_path, "w") as f:
        f.write("")
    for dataset, table in latex_tables.items():
        with open(output_path, "a") as f:
            # f.write(f"% {dataset} and {decoding_type}\n")
            f.write("\n\n\n")
            f.write(table)
            f.write("\n\n\n")


def create_s1_top_score_count_tables(
    merged_tables_all_s1: dict, ignore: list[str]
) -> dict:
    assert isinstance(ignore, list)
    s1_top_score_count_tables = {}
    for dataset, decoding_type_tables in merged_tables_all_s1.items():
        top_scores_table = None
        for _, table in decoding_type_tables.items():
            table = table.drop(ignore, axis=0)
            assert all(ign not in table.index for ign in ignore)
            if top_scores_table is None:
                index = table.index
                columns = list(set([metric_name for _, metric_name in table.columns]))
                top_scores_table = pd.DataFrame(
                    np.zeros((len(index), len(columns))), index=index, columns=columns
                )
            for model_name, metric_name in table.columns:
                if table[(model_name, metric_name)].isna().sum() == len(table):
                    continue
                else:
                    idx_max = table[(model_name, metric_name)].idxmax(
                        axis=0, skipna=True
                    )
                    top_scores_table.loc[idx_max, metric_name] += 1
        s1_top_score_count_tables[dataset] = top_scores_table
    return s1_top_score_count_tables


def reorder_s1_top_score_count_tables(
    s1_top_score_count_tables: dict, metric_order: list[str]
) -> dict:
    reordered_s1_top_score_count_tables = {}
    for dataset, top_scores_table in s1_top_score_count_tables.items():
        reordered_s1_top_score_count_tables[dataset] = top_scores_table.loc[
            :, metric_order
        ]
    return reordered_s1_top_score_count_tables


def get_s1_top_score_caption(metrics: list[str], dataset: str) -> str:
    metrics_in_caption = []
    if any(r in metrics for r in ["R-1", "R-2", "R-L"]):
        metrics_in_caption.append("ROUGE")
    if "METEOR" in metrics:
        metrics_in_caption.append("METEOR")
    if "BERTScore" in metrics:
        metrics_in_caption.append("BERTScore")
    metrics_in_caption[-1] = f"and {metrics_in_caption[-1]}"
    metrics_str_caption = ", ".join(metrics_in_caption)
    return (
        f"Number of times each pragmatic summarizer achieves the highest {metrics_str_caption} scores on the {DATASET_MAPPING[dataset]} dataset. "
        + f"We aggregate counts across all combinations of models ({' and '.join(MODEL_NAME_MAPPING.values())}) and decoding types (Beam search along with standard, nucleus, and top-k sampling)."
    )


def convert_s1_top_score_count_tables_to_latex(
    s1_top_score_count_tables: dict,
    metrics: list[str],
    add_percentage: bool = False,
    linewidth: float = 0.5,
) -> dict:
    latex_tables = deepcopy(s1_top_score_count_tables)
    for dataset, top_scores_table in s1_top_score_count_tables.items():
        if add_percentage:
            for metric in top_scores_table.columns:
                top_scores_table[metric] = top_scores_table[metric].apply(
                    lambda x: (
                        int(x)
                        if int(x) == 0
                        else f"{int(x)} ({x / top_scores_table[metric].sum() * 100:.2f}\%)"
                    )
                )
        tabulate_rows = tabulate(
            top_scores_table,
            tablefmt="latex_raw",
            headers="keys",
            showindex=True,
            # floatfmt=".3f", # NOTE: Assume that mark_max already did .3f
        ).split("\n")
        print(f"{dataset=} top scores")
        numbered_tabulate_rows = [f"{i}: {row}" for i, row in enumerate(tabulate_rows)]
        print("======Original========")
        print("\n".join(numbered_tabulate_rows))
        assert len(tabulate_rows) == 13
        full_length = top_scores_table.shape[1]
        # Replacements
        if not add_percentage:
            tabulate_rows[0] = tabulate_rows[0].replace(
                "l" + ("r" * full_length), "l" + ("c" * full_length)
            )
        else:
            tabulate_rows[0] = tabulate_rows[0].replace(
                "l" * (full_length + 1), "l" + ("c" * full_length)
            )
        tabulate_rows[1] = "\\toprule"
        tabulate_rows[-2] = "\\bottomrule"
        cmidrule_line = "\cmidrule(r){{{}-{}}}".format(2, 1 + full_length)
        tabulate_rows[3] = cmidrule_line
        tabulate_rows[4] = tabulate_rows[4] + cmidrule_line
        tabulate_rows[5] = tabulate_rows[5] + cmidrule_line
        # Additions outside table
        tabulate_rows.insert(0, "\\resizebox{" f"{linewidth}" + "\linewidth}{!}{")
        tabulate_rows.insert(0, "\centering")
        tabulate_rows.insert(0, "\\begin{table}[H]")
        tabulate_rows.append("}")
        caption = get_s1_top_score_caption(metrics=metrics, dataset=dataset)
        tabulate_rows.append("\caption{" + caption + "}")
        tabulate_rows.append(
            "\label{" + f"tab:{dataset}_prag_summ_top_score_freq" + "}"
        )
        tabulate_rows.append("\end{table}")
        print("======Modified========")
        print("\n".join(tabulate_rows))
        latex_tables[dataset] = "\n".join(tabulate_rows)
    return latex_tables


def create_relative_to_random_scores_tables(
    merged_tables_all_s1: dict,
    relative_to: str,
    ignore_index: list[str],
    ignore_decoding_type: list[str],
) -> dict:
    assert isinstance(ignore_index, list)
    assert isinstance(ignore_decoding_type, list)
    create_relative_to_random_scores_tables = {}
    for dataset, decoding_type_tables in merged_tables_all_s1.items():
        for decoding_type, table in decoding_type_tables.items():
            if decoding_type in ignore_decoding_type:
                continue
            table = table.drop(ignore_index, axis=0)
            relative_to_table = pd.DataFrame(
                np.zeros_like(table), index=table.index, columns=table.columns
            )
            for column in table.columns:
                relative_to_table[column] += (
                    table[column] / table.loc[relative_to, column]
                ) - 1
        relative_to_table = relative_to_table.drop(relative_to, axis=0)
        create_relative_to_random_scores_tables[dataset] = relative_to_table / (
            len(decoding_type_tables.keys()) - len(ignore_decoding_type)
        )
        create_relative_to_random_scores_tables[dataset] *= 100
        print(f"{dataset=} relative to random")
        print(create_relative_to_random_scores_tables[dataset])
    return create_relative_to_random_scores_tables


def reorder_relative_to_random_scores_tables(
    relative_to_random_scores_tables: dict,
    model_order: list[str],
    metric_order: list[str],
) -> dict:
    reordered_relative_to_random_scores_tables = {}
    for (
        dataset,
        relative_to_random_scores_table,
    ) in relative_to_random_scores_tables.items():
        ordered_columns = list(product(model_order, metric_order))
        reordered_relative_to_random_scores_tables[dataset] = (
            relative_to_random_scores_table[ordered_columns]
        )
    return reordered_relative_to_random_scores_tables


def get_relative_to_random_caption(metrics: list[str], dataset: str) -> str:
    metrics_in_caption = []
    if all(r in metrics for r in ["R-1", "R-2", "R-L"]):
        metrics_in_caption.append("ROUGE")
    if sum(r in metrics for r in ["R-1", "R-2", "R-L"]) < 3:
        metrics_in_caption += [
            ROUGE_NAME_MAPPING[r] for r in metrics if r in ["R-1", "R-2", "R-L"]
        ]
    if "METEOR" in metrics:
        metrics_in_caption.append("METEOR")
    if "BERTScore" in metrics:
        metrics_in_caption.append("BERTScore")
    metrics_in_caption[-1] = f"and {metrics_in_caption[-1]}"
    metrics_str_caption = ", ".join(metrics_in_caption)
    return (
        f"Relative change in {metrics_str_caption} scores between the pragmatic summarizers and $\pragrandom$ on the {DATASET_MAPPING[dataset]} dataset. "
        + "Relative changes are aggregated across all decoding methods. "
        + "A value of $+x$ ($-x$) indicates that the average metric score achieved is $x\%$ higher (lower) than the one achieved by $\pragrandom$. "
        + "The lowest relative change is \\underline{underlined}, and the highest is \\textbf{bolded}."
    )


def convert_relative_to_random_scores_tables_to_latex(
    relative_to_random_scores_tables: dict,
    metric_order: list[str],
    model_order: list[str],
    linewidth: float = 0.5,
) -> dict:
    latex_tables = deepcopy(relative_to_random_scores_tables)
    for dataset, top_scores_table in relative_to_random_scores_tables.items():
        # Add +/- explicitly
        # Underline min and bold the best
        marked_table = top_scores_table.copy(deep=True)
        for col in top_scores_table.columns:
            # Mark max with bold and the rest with +/-
            marked_table[col] = top_scores_table[col].apply(
                lambda x: (
                    ("$+$" if x > 0 else "$-$")
                    + (
                        f"\\textbf{{{abs(x):.2f}}}"
                        if x == top_scores_table[col].max()
                        else f"\\underline{{{abs(x):.2f}}}"
                    )
                    if x in [top_scores_table[col].max(), top_scores_table[col].min()]
                    else ("$+$" if x > 0 else "$-$") + f"{abs(x):.2f}"
                )
            )
        tabulate_rows = tabulate(
            marked_table,
            tablefmt="latex_raw",
            headers="keys",
            showindex=True,
            # floatfmt=".3f", # NOTE: Assume that mark_max already did .3f
        ).split("\n")
        print(f"{dataset=} top scores")
        numbered_tabulate_rows = [f"{i}: {row}" for i, row in enumerate(tabulate_rows)]
        print("======Original========")
        print("\n".join(numbered_tabulate_rows))
        assert len(tabulate_rows) == 12
        full_length = top_scores_table.shape[1]
        half_length = int(full_length / 2)
        # Replacements
        tabulate_rows[0] = tabulate_rows[0].replace(
            "l" * (full_length + 1), "l" + ("c" * full_length)
        )
        tabulate_rows[1] = "\\toprule"
        tabulate_rows[3] = ""
        tabulate_rows[-2] = "\\bottomrule"
        tabulate_rows[2] = (
            f"& \multicolumn{{{half_length}}}{{c}}{{{model_order[0]}}} & \multicolumn{{{half_length}}}{{c}}{{{model_order[1]}}} \\\\"
        )
        cmidrule_line = "\cmidrule(r){{{}-{}}}\cmidrule(r){{{}-{}}}".format(
            2, 1 + half_length, 2 + half_length, 1 + 2 * half_length
        )
        tabulate_rows[4] = tabulate_rows[4] + cmidrule_line
        tabulate_rows[6] = tabulate_rows[6] + cmidrule_line
        # Insertions within table
        tabulate_rows.insert(
            3,
            " & "
            + " & ".join([metric_name for _, metric_name in marked_table.columns])
            + " \\\\",
        )
        tabulate_rows.insert(4, cmidrule_line)
        # Additions outside table
        tabulate_rows.insert(0, "\\resizebox{" f"{linewidth}" + "\linewidth}{!}{")
        tabulate_rows.insert(0, "\centering")
        tabulate_rows.insert(0, "\\begin{table}[H]")
        tabulate_rows.append("}")
        caption = get_relative_to_random_caption(metrics=metric_order, dataset=dataset)
        tabulate_rows.append("\caption{" + caption + "}")
        tabulate_rows.append(
            "\label{" + f"tab:{dataset}_relative_to_random_scores" + "}"
        )
        tabulate_rows.append("\end{table}")
        print("======Modified========")
        print("\n".join(tabulate_rows))
        latex_tables[dataset] = "\n".join(tabulate_rows)
    return latex_tables


def read_merged_eval_and_scores_tables():
    correlations_directory = SCRATCH_DIR / "rsasumm" / "evaluation" / "correlations"
    merged_tables = {}
    for model_name in os.listdir(correlations_directory):
        merged_tables[model_name] = {}
        for decoding_type in os.listdir(
            os.path.join(correlations_directory, model_name)
        ):
            merged_tables[model_name][decoding_type] = {}
            for dataset in DATASETS:
                df = pd.read_csv(
                    os.path.join(
                        correlations_directory,
                        model_name,
                        decoding_type,
                        f"{dataset}.csv",
                    ),
                    header=[0, 1],
                    index_col=0,
                )
                df.columns = pd.MultiIndex.from_tuples(
                    [("", col[1]) if "Unnamed" in col[0] else col for col in df.columns]
                )
                merged_tables[model_name][decoding_type][dataset] = df
    return merged_tables


def compute_correlation_values(x, y, correlation, log_scale=False, normalize=False):
    if normalize:
        x = stats.zscore(x)
        y = stats.zscore(y)
    if correlation == "pearson":
        return stats.pearsonr(x, y).statistic
    elif correlation == "spearman":
        return stats.spearmanr(x, y).statistic
    elif correlation == "kendall":
        return stats.kendalltau(x, y).correlation
    elif correlation == "linreg":
        if log_scale:
            x = np.log(-1 * x)
        return stats.linregress(x, y).slope


def compute_correlations(
    eval_and_scores_table: pd.DataFrame, grouping: bool = True, log_scale: bool = False, normalize: bool = False
) -> pd.DataFrame:
    correlations = ["pearson", "spearman", "kendall"]
    correlations = ["kendall"]
    metrics = [
        x
        for x in eval_and_scores_table.columns.get_level_values(0).unique()
        if len(x) > 0
    ]
    correlations_tables = pd.DataFrame(
        columns=pd.MultiIndex.from_product([correlations, metrics])
    )
    for correlation in correlations:
        for metric in metrics:
            metric_values = eval_and_scores_table[metric]["metric_value"]
            s1_scores = eval_and_scores_table[metric]
            for s1_type in s1_scores.columns:
                if s1_type == "metric_value":
                    continue
                if grouping:
                    temp_table = pd.DataFrame(
                        {
                            "group_ids": eval_and_scores_table[("", "group_ids")],
                            "metric_value": metric_values,
                            "s1_score": s1_scores[s1_type],
                        }
                    )
                    grouped_temp_table = [
                        table for _, table in temp_table.groupby("group_ids")
                    ]
                    correlation_values = []
                    for group_table in grouped_temp_table:
                        correlation_values.append(
                            compute_correlation_values(
                                x=group_table["s1_score"],
                                y=group_table["metric_value"],
                                correlation=correlation,
                                log_scale=log_scale,
                                normalize=normalize,
                            )
                        )
                    correlations_tables.loc[s1_type, (correlation, metric)] = np.mean(
                        correlation_values
                    )
                else:
                    correlations_tables.loc[s1_type, (correlation, metric)] = (
                        compute_correlation_values(
                            x=s1_scores[s1_type],
                            y=metric_values,
                            correlation=correlation,
                            log_scale=log_scale,
                            normalize=normalize,
                        )
                    )
                fig = px.scatter(
                    x=stats.zscore(s1_scores[s1_type]),
                    y=stats.zscore(metric_values),
                    title=f"{metric} vs {s1_type}",
                )
                fig.write_image("scatter_plot.png")
            # breakpoint()
    return correlations_tables


def compute_correlations_tables(merged_eval_and_scores_tables: dict):
    correlations_tables = {}
    for model_name, decoding_types in merged_eval_and_scores_tables.items():
        correlations_tables[model_name] = {}
        for decoding_type, datasets in decoding_types.items():
            correlations_tables[model_name][decoding_type] = {}
            for dataset, df in datasets.items():
                print(f"{model_name=} {decoding_type=} {dataset=}")
                correlations_tables[model_name][decoding_type][dataset] = (
                    compute_correlations(df, grouping=True, log_scale=False, normalize=False)
                )
                print(correlations_tables[model_name][decoding_type][dataset])
    return correlations_tables


@click.command()
@click.option(
    "--eval_yaml_path",
    help="Dataset name",
    type=str,
    required=True,
    default=SRC_DIRECTORY / "evaluation" / "evaluations" / "llama_bart_qfs.yaml",
)
@click.option(
    "--table_type",
    help="Table type",
    required=True,
    type=click.Choice(
        [
            "table_all_s1",
            "s1_top_score_count",
            "s1_top_score_count_aggregated_all",
            "s1_relative_to_random_scores",
            "s1_relative_to_random_scores_counts",
            "correlation_table",
        ]
    ),
)
@click.option(
    "--output_dir",
    help="Output directory",
    type=str,
    required=True,
    default=SCRATCH_DIR / "rsasumm" / "paper_tables",
)
@click.option(
    "--file_name",
    help="Table name",
    type=str,
    required=False,
)
def main(
    eval_yaml_path: str, table_type: str, output_dir: str, file_name: str = None
) -> None:
    evals = get_evals(eval_yaml_path=eval_yaml_path)
    metric_order = ["Comp.", "Rep.", "Gram.", "Conc."]
    metric_order = ["R-1", "R-2", "R-L", "METEOR", "BERTScore"]
    model_order = ["BART", "Llama 3"]
    if table_type == "table_all_s1":
        tables_all_s1 = create_tables_all_s1(evals=evals)
        merged_tables_all_s1 = merge_tables_all_s1_by_decoding_type(
            tables_all_s1=tables_all_s1
        )
        s1_order = list(S1_NAME_MAPPING.values())
        oracle = s1_order.pop(0)
        s1_order.append(oracle)
        reformated_tables_all_s1 = reformat_tables_all_s1(
            tables_all_s1=merged_tables_all_s1,
            model_order=model_order,
            metric_order=metric_order,
            s1_order=s1_order,
        )
        latex_tables = convert_tables_all_s1_to_latex(
            reformated_tables_all_s1, model_order=model_order, metric_order=metric_order
        )
        write_all_s1_latex_tables(
            latex_tables=latex_tables,
            file_name=table_type if file_name is None else file_name,
            output_dir=output_dir,
            skip=["standard_beam"],
        )
    elif table_type == "s1_top_score_count":
        tables_all_s1 = create_tables_all_s1(evals=evals)
        merged_tables_all_s1 = merge_tables_all_s1_by_decoding_type(
            tables_all_s1=tables_all_s1
        )
        s1_top_score_count_tables = create_s1_top_score_count_tables(
            merged_tables_all_s1=merged_tables_all_s1,
            ignore=["$\pragoracle$"],
        )
        s1_top_score_count_tables = reorder_s1_top_score_count_tables(
            s1_top_score_count_tables=s1_top_score_count_tables,
            metric_order=metric_order,
        )
        latex_tables = convert_s1_top_score_count_tables_to_latex(
            s1_top_score_count_tables=s1_top_score_count_tables,
            metrics=metric_order,
            add_percentage=False,
            linewidth=0.5,
        )
        write_s1_top_score_count_latex_tables(
            latex_tables=latex_tables,
            file_name=table_type if file_name is None else file_name,
            output_dir=output_dir,
        )
    elif table_type == "s1_top_score_count_aggregated_all":
        tables_all_s1 = create_tables_all_s1(evals=evals)
        merged_tables_all_s1 = merge_tables_all_s1_by_decoding_type(
            tables_all_s1=tables_all_s1
        )
        s1_top_score_count_tables = create_s1_top_score_count_tables(
            merged_tables_all_s1=merged_tables_all_s1,
            ignore=["$\pragoracle$"],
        )
        s1_top_score_count_tables = reorder_s1_top_score_count_tables(
            s1_top_score_count_tables=s1_top_score_count_tables,
            metric_order=metric_order,
        )
        s1_top_score_count_agg_all = None
        for dataset, top_scores_table in s1_top_score_count_tables.items():
            if s1_top_score_count_agg_all is None:
                s1_top_score_count_agg_all = pd.DataFrame(
                    np.zeros(
                        (len(top_scores_table), len(s1_top_score_count_tables.keys()))
                    ),
                    index=top_scores_table.index,
                    columns=list(s1_top_score_count_tables.keys()),
                )
            s1_top_score_count_agg_all[dataset] = top_scores_table.sum(axis=1)
        s1_top_score_count_agg_all["total"] = s1_top_score_count_agg_all.sum(axis=1)
        s1_top_score_count_agg_all["percentage"] = (
            s1_top_score_count_agg_all["total"]
            / s1_top_score_count_agg_all["total"].sum()
            * 100
        )
        print(s1_top_score_count_agg_all)
    elif table_type == "s1_relative_to_random_scores":
        # NOTE: This is ignoring standard beam search
        tables_all_s1 = create_tables_all_s1(evals=evals)
        merged_tables_all_s1 = merge_tables_all_s1_by_decoding_type(
            tables_all_s1=tables_all_s1
        )
        relative_to_random_scores_tables = create_relative_to_random_scores_tables(
            merged_tables_all_s1=merged_tables_all_s1,
            relative_to="$\pragrandom$",
            ignore_index=["$\pragoracle$"],
            ignore_decoding_type=["standard_beam"],
        )
        relative_to_random_scores_tables = reorder_relative_to_random_scores_tables(
            relative_to_random_scores_tables=relative_to_random_scores_tables,
            model_order=model_order,
            metric_order=metric_order,
        )
        latex_tables = convert_relative_to_random_scores_tables_to_latex(
            relative_to_random_scores_tables=relative_to_random_scores_tables,
            model_order=model_order,
            metric_order=metric_order,
            linewidth=1,
        )
        write_s1_top_score_count_latex_tables(
            latex_tables=latex_tables,
            file_name=table_type if file_name is None else file_name,
            output_dir=output_dir,
        )
    elif table_type == "s1_relative_to_random_scores_counts":
        tables_all_s1 = create_tables_all_s1(evals=evals)
        merged_tables_all_s1 = merge_tables_all_s1_by_decoding_type(
            tables_all_s1=tables_all_s1
        )
        relative_to_random_scores_tables = create_relative_to_random_scores_tables(
            merged_tables_all_s1=merged_tables_all_s1,
            relative_to="$\pragrandom$",
            ignore_index=["$\pragoracle$"],
            ignore_decoding_type=["standard_beam"],
        )
        relative_to_random_scores_tables = reorder_relative_to_random_scores_tables(
            relative_to_random_scores_tables=relative_to_random_scores_tables,
            model_order=model_order,
            metric_order=metric_order,
        )
        s1_relative_to_random_scores_counts = None
        for dataset, top_scores_table in relative_to_random_scores_tables.items():
            if s1_relative_to_random_scores_counts is None:
                s1_relative_to_random_scores_counts = pd.DataFrame(
                    np.zeros(
                        (
                            top_scores_table.shape[0],
                            len(list(relative_to_random_scores_tables.keys())) + 1,
                        )
                    ),
                    index=top_scores_table.index,
                    columns=list(relative_to_random_scores_tables.keys()) + ["total"],
                )
            s1_relative_to_random_scores_counts[dataset] += (top_scores_table > 0).sum(
                axis=1
            )
            s1_relative_to_random_scores_counts["total"] += top_scores_table.shape[1]
            s1_relative_to_random_scores_counts["percentage"] = (
                s1_relative_to_random_scores_counts[DATASETS].sum(axis=1)
                / s1_relative_to_random_scores_counts["total"]
                * 100
            )
        print(s1_relative_to_random_scores_counts)
    elif table_type == "correlation_table":
        merged_eval_and_scores_tables = read_merged_eval_and_scores_tables()
        correlation_tables = compute_correlations_tables(merged_eval_and_scores_tables)
    else:
        raise ValueError(f"Table type {table_type} not recognized.")


if __name__ == "__main__":
    main()
