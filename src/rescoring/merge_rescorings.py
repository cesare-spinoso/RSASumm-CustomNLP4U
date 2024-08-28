import os

import hydra
import pandas as pd
from omegaconf import DictConfig

from src import DATASET_NAMES, SRC_DIRECTORY
from src.rescoring.rescore_summaries import process_llama3_preds
from src.utils.dataset import get_question_column_name
from src.utils.decorators import main_decorator
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines


def sanity_check_config(cfg):
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
                "qfs_summaries",
                "source_reconstruction",
                "latent_reconstruction",
                "answer_reconstruction",
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


def merge_rescored_qfs_summaries(dataset_name, cfg):
    print("=" * 60)
    print(f"Merging for {dataset_name} summaries.")
    # Read the csv file
    dataset_path = cfg["datasets"][dataset_name]
    df_raw = pd.read_csv(dataset_path)
    question_column_name = get_question_column_name(dataset_name)
    df_raw = df_raw[["document", question_column_name, "summary"]]
    df_raw.rename(
        columns={"document": "source", "summary": "reference_summary"}, inplace=True
    )
    print(f"{len(df_raw)=}")
    # Get the summary df
    ordered_keys = [cfg["summarizer_name"], dataset_name]
    yaml_path = cfg["yaml_files"]["qfs_summaries"]
    jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
    summary_jsonlines = read_jsonlines(jsonlines_path)
    df_generated_summary = pd.DataFrame(summary_jsonlines)
    df_generated_summary = df_generated_summary[
        ["source", question_column_name, "pred", "pred_score"]
    ]
    df_generated_summary.rename(columns={"pred": "generated_summary"}, inplace=True)
    if "llama3" in cfg["summarizer_name"]:
        df_generated_summary["generated_summary"] = process_llama3_preds(
            df_generated_summary["generated_summary"].tolist(), pred_type="summary"
        )
    print(f"{len(df_generated_summary)=}")
    # Merge the generated summary with the raw data
    df_merged = pd.merge(
        df_raw, df_generated_summary, on=["source", question_column_name]
    )
    print(f"{len(df_merged)=}")
    # Get the answer reconstruction scores
    if "answer_reconstruction" in cfg["yaml_files"]:
        ordered_keys = [cfg["summarizer_name"], dataset_name]
        yaml_path = cfg["yaml_files"]["answer_reconstruction"]
        jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
        answer_rec_jsonlines = read_jsonlines(jsonlines_path)
        df_answer_rec = pd.DataFrame(answer_rec_jsonlines)
        df_answer_rec = df_answer_rec[
            [
                "source",
                question_column_name,
                "summary",
                "answer",
                "reconstruction_score",
                "avg_reconstruction_score",
            ]
        ]
        df_answer_rec.rename(
            columns={
                "summary": "generated_summary",
                "answer": "generated_answer",
                "reconstruction_score": "ans_rec_score",
                "avg_reconstruction_score": "avg_ans_rec_score",
            },
            inplace=True,
        )
        print(f"{len(df_answer_rec)=}")
        # Merge the answer reconstruction scores with the generated summary
        df_merged = pd.merge(
            df_merged,
            df_answer_rec,
            on=["source", question_column_name, "generated_summary"],
        )
        print(f"{len(df_merged)=}")
    # Get the source reconstruction scores
    if "source_reconstruction" in cfg["yaml_files"]:
        ordered_keys = [cfg["summarizer_name"], dataset_name]
        yaml_path = cfg["yaml_files"]["source_reconstruction"]
        jsonlines_path = get_jsonl_path_from_yaml(ordered_keys, yaml_path)
        source_rec_jsonlines = read_jsonlines(jsonlines_path)
        df_source_rec = pd.DataFrame(source_rec_jsonlines)
        start_marker = "Do not provide anything else other than the reconstructed source document. Summary:"
        end_marker = "Source: <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        assert (
            df_source_rec["pred"].apply(lambda x: len(x.split(start_marker)) == 2).all()
        )
        assert (
            df_source_rec["pred"].apply(lambda x: len(x.split(end_marker)) == 2).all()
        )
        df_source_rec["pred"] = df_source_rec["pred"].apply(
            lambda x: x.split(start_marker)[-1].split(end_marker)[0].strip()
        )
        df_source_rec = df_source_rec[
            ["source", "pred", "reconstruction_score", "reconstruction_score"]
        ]
        df_source_rec.rename(
            columns={
                "pred": "generated_summary",
                "reconstruction_score": "source_rec_score",
                "avg_reconstruction_score": "source_avg_rec_score",
            },
            inplace=True,
        )
        assert df_source_rec["source"].isin(df_merged["source"]).all()
        df_merged["generated_summary"] = df_merged["generated_summary"].apply(
            lambda x: x.strip()
        )
        assert (
            df_source_rec["generated_summary"]
            .isin(df_merged["generated_summary"])
            .all()
        )
        df_merged = pd.merge(
            df_merged,
            df_source_rec,
            on=["source", "generated_summary"],
        )
        print(f"{len(df_merged)=}")
    return df_merged


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf", "merge_rescores"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    sanity_check_config(cfg)
    for dataset_name in cfg["datasets"]:
        if "qfs" in cfg["summarizer_name"]:
            df = merge_rescored_qfs_summaries(dataset_name=dataset_name, cfg=cfg)
        else:
            raise ValueError(f"Unknown summarizer name: {cfg['summarizer_name']}")
        os.makedirs(os.path.join(cfg["output_directory"], run_name), exist_ok=True)
        df.to_csv(
            os.path.join(cfg["output_directory"], run_name, f"{dataset_name}.csv"),
            index=False,
        )
        df = df.map(
            lambda x: (
                x.encode("unicode_escape").decode("utf-8") if isinstance(x, str) else x
            )
        )
        df.to_excel(
            os.path.join(cfg["output_directory"], run_name, f"{dataset_name}.xlsx"),
            index=False,
        )


if __name__ == "__main__":
    main()
