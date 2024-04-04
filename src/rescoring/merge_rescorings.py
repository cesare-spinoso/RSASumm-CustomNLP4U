from collections import OrderedDict
import os

import hydra
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from src import DATASET_NAMES, SRC_DIRECTORY
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.decorators import main_decorator
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines


def sanity_check_config(cfg):
    hydra_job = HydraConfig.get().job
    hydra_runtime = HydraConfig.get().runtime
    assert cfg["summarizer_name"] == hydra_job["config_name"]
    assert any(
        "merge_rescores" in dict_elt["path"]
        for dict_elt in hydra_runtime["config_sources"]
    )
    assert (
        isinstance(cfg["datasets"], list) and set(cfg["datasets"]) <= DATASET_NAMES
    ) or (
        isinstance(cfg["datasets"], dict)
        and set(cfg["datasets"].keys()) <= DATASET_NAMES
    )
    assert all(
        any(
            k.startswith(s)
            for s in [
                "generic_summaries",
                "e2e_summaries",
                "source_reconstruction",
                "latent_reconstruction",
            ]
        )
        for k in cfg["yaml_files"]
    )
    assert all(
        "merge_rescores" in cfg[k]
        for k in ["write_config_directory", "output_directory"]
    )


def rename_columns(df, name):
    if name == "source_reconstruction":
        df = df.rename(
            columns={
                "reconstruction_score": "source_rec_score",
                "avg_reconstruction_score": "source_avg_rec_score",
            }
        )
    elif name == "latent_reconstruction":
        df = df.rename(
            columns={
                "reconstruction_score": "latent_rec_score",
                "avg_reconstruction_score": "latent_avg_rec_score",
            }
        )
    return df


def merge_rescored_generic_summaries(dataset_name, cfg):
    # Preprocess input needed to compute metrics
    ordered_keys = [cfg["summarizer_name"], dataset_name]
    # Read jsonlines data
    dfs = OrderedDict()
    for name, yaml_path in cfg["yaml_files"].items():
        jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
        jsonlines_data = read_jsonlines(jsonlines_path)
        df = pd.DataFrame(jsonlines_data)
        # NOTE: Some  predictions are "", we still use them for the merge
        if cfg["rename_columns"]:
            df = rename_columns(df, name)
        dfs[name] = df
    # Merge three dataframes
    merged_df = None
    for df in dfs.values():
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=cfg["columns_to_merge_on"])
    # Drop any nan's introduced, e.g. because pred is empty
    merged_df = merged_df.dropna()
    return merged_df


def merge_rescored_e2e_summaries(dataset_name, dataset_path, cfg):
    # Read the csv file
    df_raw = pd.read_csv(dataset_path)
    latent_column_name = get_latent_column_name(dataset_name)
    # Get the summary df and add the latent column to it
    # from the question
    ordered_keys = [cfg["summarizer_name"], dataset_name]
    yaml_path = cfg["yaml_files"]["e2e_summaries"]
    jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
    summary_jsonlines = read_jsonlines(jsonlines_path)
    df_summary = pd.DataFrame(summary_jsonlines)
    if dataset_name == "covidet":
        df_raw["question"] = df_raw["emotion"].apply(
            lambda x: cfg["covidet_template"].format(emotion=x)
        )
    elif dataset_name == "debatepedia":
        df_raw["question"] = df_raw["query"]
    question_to_latent = dict(
        zip(
            df_raw["question"].tolist(),
            df_raw[latent_column_name],
        )
    )
    latent = [question_to_latent[q] for q in df_summary["question"]]
    df_summary[latent_column_name] = latent
    df_summary = df_summary[
        ["document_id", "source", "pred", "pred_score", latent_column_name]
    ]
    # Merge the reconstruction scores from source
    source_yaml_path = cfg["yaml_files"]["source_reconstruction"]
    source_jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, source_yaml_path)
    source_rec_jsonlines = read_jsonlines(source_jsonlines_path)
    input_to_score = {
        (elt["source"], elt["pred"]): elt["reconstruction_score"]
        for elt in source_rec_jsonlines
    }
    input_to_avg_score = {
        (elt["source"], elt["pred"]): elt["avg_reconstruction_score"]
        for elt in source_rec_jsonlines
    }
    df_summary["source_rec_score"] = df_summary.apply(
        lambda x: input_to_score[(x["source"], x["pred"])], axis=1
    )
    df_summary["avg_source_rec_score"] = df_summary.apply(
        lambda x: input_to_avg_score[(x["source"], x["pred"])], axis=1
    )
    # Merge the latent reconstruction
    # NOTE: The pred score for the same summary can be different
    # so merged_df may have length > len(df_summary) or len(df_latent)
    latent_yaml_path = cfg["yaml_files"]["latent_reconstruction"]
    jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, latent_yaml_path)
    latent_rec_jsonlines = read_jsonlines(jsonlines_path)
    df_latent = pd.DataFrame(latent_rec_jsonlines)
    df_latent = rename_columns(df_latent, "latent_reconstruction")
    merged_df = pd.merge(
        df_summary, df_latent, on=["document_id", "source", "pred", latent_column_name]
    )
    return merged_df


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf", "merge_rescores"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    for dataset_name in cfg["datasets"]:
        if "generic" in cfg["summarizer_name"]:
            df = merge_rescored_generic_summaries(dataset_name=dataset_name, cfg=cfg)
        elif "e2e" in cfg["summarizer_name"]:
            df = merge_rescored_e2e_summaries(
                dataset_name=dataset_name,
                dataset_path=cfg["datasets"][dataset_name],
                cfg=cfg,
            )
        else:
            raise ValueError(f"Unknown summarizer name: {cfg['summarizer_name']}")
        os.makedirs(os.path.join(cfg["output_directory"], run_name), exist_ok=True)
        df.to_csv(
            os.path.join(cfg["output_directory"], run_name, f"{dataset_name}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
