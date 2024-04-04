import json
from ast import literal_eval as make_tuple

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots
from tabulate import tabulate

# Dataframes


def convert_eval_json_to_df(json_path, evals_to_exclude=None):
    if evals_to_exclude is None:
        evals_to_exclude = []
    with open(json_path, "r") as f:
        data = json.load(f)
    data = {
        ds_name: {
            eval_name: f"{eval_dict['rouge1']:.3f}/{eval_dict['rouge2']:.3f}/{eval_dict['rougeL']:.3f}"
            for eval_name, eval_dict in eval_dict.items()
            if eval_name not in evals_to_exclude
        }
        for ds_name, eval_dict in data.items()
    }
    return pd.DataFrame(data)


def metric_dict_to_string(metric_dict, diversity_to_exclude):
    assert len(metric_dict.keys()) and list(metric_dict.keys())[0] in ["Hamming", "LCS"]
    key_ = list(metric_dict.keys())[0]
    metrics = metric_dict[key_]
    return " / ".join(
        [f"{v:.3f}" for k, v in metrics.items() if k not in diversity_to_exclude]
    )


def convert_diversity_json_to_df(json_path, diversity_to_exclude=None):
    if diversity_to_exclude is None:
        diversity_to_exclude = []
    with open(json_path, "r") as f:
        data = json.load(f)
    data = {
        model_name: {
            dataset_name: metric_dict_to_string(metric_dict, diversity_to_exclude)
            for dataset_name, metric_dict in dataset_dict.items()
        }
        for model_name, dataset_dict in data.items()
    }
    return pd.DataFrame(data)


def convert_weighted_rsa_json_to_dfs(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    dfs = {}
    for ds_name, eval_dict in data.items():
        weighted_rsa_dict = eval_dict["weighted_rsa"]
        keys = [make_tuple(s) for s in weighted_rsa_dict.keys()]
        lambdas = sorted(list(set([k[0] for k in keys])))
        alphas = sorted(list(set([k[1] for k in keys])))
        dfs[ds_name] = {
            metric: pd.DataFrame(None, index=lambdas, columns=alphas)
            for metric in ["rouge1", "rouge2", "rougeL"]
        }
        for k, metrics_dict in weighted_rsa_dict.items():
            lamb, alpha = make_tuple(k)
            for metric, value in metrics_dict.items():
                dfs[ds_name][metric].loc[lamb, alpha] = value
    return dfs


def pretty_print(title, df):
    if title is not None:
        print(title)
    print(tabulate(df, headers="keys", tablefmt="psql"))


# PLotly


def compute_ranges(n):
    gap = 0.05
    total_gap = gap * (n - 1)
    total_remaining = 1 - total_gap
    width = total_remaining / n
    ranges = []
    for i in range(n):
        ranges.append([i * (width + gap), i * (width + gap) + width])
    return ranges


def create_weighted_rsa_heatmap_figs(dfs):
    # Make subplot
    nrows = len(dfs)
    example_dict = list(dfs.values())[0]
    ncols = len(example_dict)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        horizontal_spacing=0.05,
        vertical_spacing=0.05,
        column_titles=[metric for metric in example_dict.keys()],
        row_titles=list(reversed([ds_name for ds_name in dfs.keys()])),
        row_heights=[1 for _ in range(nrows)],
        column_widths=[1 for _ in range(ncols)],
        x_title="alpha",
        y_title="lambda",
    )
    for i, k in enumerate(dfs.keys()):
        for j, metric in enumerate(dfs[k].keys()):
            df = dfs[k][metric]
            fig.add_trace(
                go.Heatmap(
                    x=df.columns,
                    y=df.index,
                    z=df.values,
                    colorbar_x=j / 3 + 0.29 + 0.015 * j,
                    colorbar_y=1 - (i / 5 + 0.1),
                    colorbar=dict(
                        lenmode="fraction",
                        len=0.15,
                        thickness=10,
                        tickfont=dict(size=10),
                    ),
                ),
                row=i + 1,
                col=j + 1,
            )
    fig.update_layout(
        height=1000,
        width=1600,
    )
    return fig


# Dash


def dropdown_heatmaps(fig_dict, jupyter):
    # Initialize Dash app with JupyterDash class
    if jupyter:
        app = JupyterDash(__name__)
    else:
        app = dash.Dash(__name__)

    # Define layout
    app.layout = html.Div(
        [
            dcc.Dropdown(
                id="figure-dropdown",
                options=[{"label": key, "value": key} for key in fig_dict.keys()],
                value=list(fig_dict.keys())[0],  # Set default value to the first key
            ),
            html.Div(id="figure-container"),
        ]
    )

    # Define callback to update the displayed figure
    @app.callback(
        Output("figure-container", "children"), [Input("figure-dropdown", "value")]
    )
    def update_figure(selected_figure):
        # Get the selected figure from the dictionary
        selected_fig = fig_dict[selected_figure]
        # Return the figure as a Graph component
        return dcc.Graph(figure=selected_fig)

    return app
