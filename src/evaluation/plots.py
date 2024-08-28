from itertools import product
import os
import click
import pandas as pd
from copy import deepcopy
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots


from src import SCRATCH_DIR, SRC_DIRECTORY
from src.evaluation.tables import DECODING_TYPE_MAPPING, METRIC_NAME_MAPPING, MODEL_NAME_MAPPING, get_evals
from src.utils.helper import read_json_file, read_yaml
from tabulate import tabulate

VARIABLE_LAMBDA_S1_NAME_MAPPING = {
    "weighted_rsa_source_scaled": r"S_1^{\texttt{Source}}",
    "weighted_rsa_answer_scaled": r"S_1^{\texttt{Answer}}",
}

S1_TO_PLOTLY_LINE_STYPE = {
    "weighted_rsa_answer_scaled": "solid",
    "weighted_rsa_source_scaled": "dash",
}

METRIC_TO_DASH_TYPE = {
    "rouge1": "solid",
    "meteor": "solid",
    "bertscore": "solid",
    "seahorse-comprehensible": "dot",
    "seahorse-repetition": "dot",
    "seahorse-grammar": "dot",
    "seahorse-conciseness": "dot",
}

METRIC_TO_COLOR_PLOTLY = {
    "rouge1": "#1f77b4",
    "meteor": "#d62728",
    "bertscore": "#2ca02c",
    "seahorse-comprehensible": "#1f77b4",  # Blue
    "seahorse-repetition": "#d62728",  # Red
    "seahorse-grammar": "#2ca02c",  # Green
    "seahorse-conciseness": "#ff7f0e",  # Orange
}

DATASET_NAME_MAPPING = {
    "multioped": "MultiOpEd",
    "qmsum": "QMSum",
    "squality": "SQuALITY",
}

DATASET_NAME_MAPPING_LATEX = {
    "multioped": "\multioped",
    "qmsum": "\qmsum",
    "squality": "\squality",
}


def create_metrics_versus_lambda_table(metric: str, eval: dict) -> pd.DataFrame:
    s1_metric_values = {s1: [] for s1 in VARIABLE_LAMBDA_S1_NAME_MAPPING.keys()}
    lambdas = []
    for s1 in VARIABLE_LAMBDA_S1_NAME_MAPPING.keys():
        metric_dict = eval[s1]
        for metric_value in metric_dict.values():
            s1_metric_values[s1].append(metric_value[metric])
    lambdas = list(metric_dict.keys())
    table = pd.DataFrame(data=s1_metric_values, index=lambdas)
    return table


def create_tables_metrics_versus_lambda(evals: dict):
    tables_metrics_versus_lambda = {}
    for metric, model_evals in evals.items():
        tables_metrics_versus_lambda[metric] = {}
        for model_name, decoding_types in model_evals.items():
            tables_metrics_versus_lambda[metric][model_name] = {}
            for decoding_type, datasets_eval in decoding_types.items():
                tables_metrics_versus_lambda[metric][model_name][decoding_type] = {}
                for dataset, eval in datasets_eval.items():
                    if eval is None:
                        continue
                    table = create_metrics_versus_lambda_table(metric=metric, eval=eval)
                    tables_metrics_versus_lambda[metric][model_name][decoding_type][
                        dataset
                    ] = table
    return tables_metrics_versus_lambda


def merge_metrics_versus_lambda_tables(tables_metrics_versus_lambda: dict) -> dict:
    merged_tables_metrics_versus_lambda = {}
    for metric, model_evals in tables_metrics_versus_lambda.items():
        for model_name, decoding_types in model_evals.items():
            if model_name not in merged_tables_metrics_versus_lambda:
                merged_tables_metrics_versus_lambda[model_name] = {}
            for decoding_type, datasets_eval in decoding_types.items():
                if decoding_type not in merged_tables_metrics_versus_lambda[model_name]:
                    merged_tables_metrics_versus_lambda[model_name][decoding_type] = {}
                for dataset, table in datasets_eval.items():
                    table.columns = pd.MultiIndex.from_tuples(
                        [(metric, col) for col in table.columns]
                    )
                    if (
                        dataset
                        not in merged_tables_metrics_versus_lambda[model_name][
                            decoding_type
                        ]
                    ):
                        merged_tables_metrics_versus_lambda[model_name][decoding_type][
                            dataset
                        ] = table
                    else:
                        current_table = merged_tables_metrics_versus_lambda[model_name][
                            decoding_type
                        ][dataset]
                        merged_tables_metrics_versus_lambda[model_name][decoding_type][
                            dataset
                        ] = pd.concat([current_table, table], axis=1)
    # Swap the grouping order of the MultiIndex
    for model_name, decoding_types in merged_tables_metrics_versus_lambda.items():
        for decoding_type, datasets_eval in decoding_types.items():
            for dataset, table in datasets_eval.items():
                table = table.swaplevel(0, 1, axis=1)
                table = table.sort_index(axis=1)
                merged_tables_metrics_versus_lambda[model_name][decoding_type][
                    dataset
                ] = table
    return merged_tables_metrics_versus_lambda


def plot_grouped_columns_side_by_side(
    df, filename, metric_order: list[str] = None, dataset_name: str = None
):
    # Get the unique groups from the first level of the MultiIndex
    groups = df.columns.get_level_values(0).unique()

    # Create subplots with one row and as many columns as there are groups
    fig = make_subplots(
        rows=1,
        cols=len(groups),
        shared_yaxes=True,
        shared_xaxes=True,
        subplot_titles=[
            f"$\mathrm{{{VARIABLE_LAMBDA_S1_NAME_MAPPING[group]}}}$" for group in groups
        ],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    )

    # Iterate through the groups and create a subplot for each
    for i, group in enumerate(groups):
        # Get the columns that belong to the current group
        group_columns = df[group]

        # Iterate through each column in the group
        if metric_order is None:
            columns = group_columns.columns
        else:
            columns = [m for m in metric_order if m in group_columns.columns]
        for col in columns:
            fig.add_trace(
                go.Scatter(
                    x=[round(float(x), 3) for x in df.index],
                    y=group_columns[col],
                    line=dict(
                        dash=METRIC_TO_DASH_TYPE[col],
                        color=METRIC_TO_COLOR_PLOTLY[col],
                    ),
                    name=f"{METRIC_NAME_MAPPING[col]}",
                    showlegend=(i == 0),  # Show legend only for the first group
                ),
                row=1,
                col=i + 1,
                secondary_y=col
                not in ["rouge1", "rouge2", "rougeL", "meteor", "bertscore"],
            )

    # Update layout for better visualization
    s1_in_title_str = " and ".join(
        [f"$\mathrm{{{v}}}$" for v in VARIABLE_LAMBDA_S1_NAME_MAPPING.values()]
    )
    fig.update_layout(
        # title_text=r"$\text{Summarization and Text Quality vs. }\lambda$",
        title="",
        xaxis=dict(title=r"$\lambda$", tickangle=-45),
        xaxis2=dict(title=r"$\lambda$", tickangle=-45),
        yaxis=dict(title="Summarization Quality"),
        yaxis2=dict(title="", title_standoff=2),
        yaxis3=dict(title="", showticklabels=True, title_standoff=2),
        yaxis4=dict(title="Text Quality"),
        legend_title="Metrics",
        template="plotly_white",
        showlegend=True,
    )

    # Make all subplots share a single legend
    fig.update_layout(legend=dict(x=1.1, y=0.5, tracegroupgap=0))

    # Save the figure to a file
    fig.write_image(filename)


@click.command()
@click.option(
    "--eval_yaml_path",
    help="Dataset name",
    type=str,
    required=True,
    default=SRC_DIRECTORY / "evaluation" / "evaluations" / "llama_bart_qfs.yaml",
)
@click.option(
    "--plot_type",
    help="Plot type",
    required=True,
    type=click.Choice(
        [
            "metrics_versus_lambda",
        ]
    ),
    default="metrics_versus_lambda",
)
@click.option(
    "--output_dir",
    help="Output directory",
    type=str,
    required=True,
    default=SCRATCH_DIR / "rsasumm" / "paper_plots",
)
def main(eval_yaml_path: str, plot_type: str, output_dir: str) -> None:
    evals = get_evals(eval_yaml_path=eval_yaml_path)
    if plot_type == "metrics_versus_lambda":
        tables_metrics_versus_lambda = create_tables_metrics_versus_lambda(evals=evals)
        merged_tables_metrics_versus_lambda = merge_metrics_versus_lambda_tables(
            tables_metrics_versus_lambda=tables_metrics_versus_lambda
        )
        output_dir = os.path.join(output_dir, "side_by_side", "metrics_versus_lambda")
        os.makedirs(output_dir, exist_ok=True)
        metric_order = [
            "rouge1",
            "meteor",
            "bertscore",
            "seahorse-comprehensible",
            "seahorse-repetition",
            "seahorse-grammar",
            "seahorse-conciseness",
        ]
        latex_template = """
        \\begin{{figure}}[H]
            \centering
            \includegraphics[width=\linewidth]{{{filename}}}
            \caption{{{caption}}}
            \label{{{label}}}
        \end{{figure}}
        """
        with open(os.path.join(output_dir, f"metrics_versus_lambda.tex"), "w") as f:
            f.write("")
        for model_name, decoding_types in merged_tables_metrics_versus_lambda.items():
            for decoding_type, datasets_eval in decoding_types.items():
                for dataset, table in datasets_eval.items():
                    filename = f"{model_name}_{decoding_type}_{dataset}_metrics_versus_lambda_complete.png"
                    plot_grouped_columns_side_by_side(
                        df=table,
                        filename=os.path.join(
                            output_dir,
                            filename,
                        ),
                        metric_order=metric_order,
                    )
                    label = filename.replace(".png", "")
                    caption = (
                        f"Tradeoff between summarization quality and text quality as controlled by $\lambda$ for the {DATASET_NAME_MAPPING_LATEX[dataset]} dataset using {MODEL_NAME_MAPPING[model_name]} with {DECODING_TYPE_MAPPING[decoding_type].lower()}. "
                        + f"Solid lines represent summarization quality metrics (ROUGE-1, METEOR, and BERTScore), while dashed lines represent text quality metrics (Comprehensibility, Repetition, Conciseness, and ). "
                        + f"Note that the left and right y-axes have different scales for summarization quality and text quality, respectively. "
                    )
                    with open(
                        os.path.join(output_dir, f"metrics_versus_lambda.tex"),
                        "a",
                    ) as f:
                        f.write(
                            latex_template.format(
                                filename="figs/" + filename,
                                caption=caption,
                                label=label,
                            )
                        )
    else:
        raise ValueError(f"Table type {plot_type} not recognized.")


if __name__ == "__main__":
    main()
