import hydra
import os
import jsonlines
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from tabulate import tabulate
from src.utils.decorators import main_decorator
from src import SRC_DIRECTORY


def compute_reranked_scores(df: pd.DataFrame, eval_info: dict) -> pd.DataFrame:
    # Calculate scores by different rankings
    score_types = eval_info["score_type"]
    assert isinstance(score_types, list)
    num_seq_per_source = eval_info["num_seq_per_source"]
    ranking_types = eval_info["ranking_types"]
    avg_scores = {}
    for score_type in score_types:
        avg_scores[score_type] = compute_avg_scores(
            df, score_type, num_seq_per_source, ranking_types
        )
    df = pd.DataFrame(avg_scores)
    return df


def compute_reranked_frequencies(df: pd.DataFrame, eval_info: dict) -> pd.DataFrame:
    # Calculate score frequencies by different rankings
    score_types = eval_info["score_type"]
    assert isinstance(score_types, list)
    num_seq_per_source = eval_info["num_seq_per_source"]
    ranking_types = eval_info["ranking_types"]
    frequencies = {}
    for score_type in score_types:
        frequencies[score_type] = compute_frequencies(
            df, score_type, num_seq_per_source, ranking_types
        )
    df = pd.DataFrame(frequencies)
    return df


def compute_avg_scores(
    df, score_type, num_seq_per_source, ranking_types
) -> dict[str, float]:
    scores = {ranking_type: [] for ranking_type in ranking_types}
    for i in range(0, len(df), num_seq_per_source):
        df_slice = df.loc[i : i + num_seq_per_source - 1]
        for ranking_type, json_key_column in ranking_types.items():
            slice_scores = df_slice[score_type].values
            if ranking_type == "random":
                scores[ranking_type].append(np.random.choice(slice_scores, size=1)[0])
            elif ranking_type == "max":
                scores[ranking_type].append(slice_scores.max())
            elif ranking_type in "direct_score":
                # The direct scores are the accumlated logits e.g. T5 model
                # The highest score is the best according to this decoding method
                ranking_scores = df_slice[json_key_column].values
                top_score = slice_scores[(-ranking_scores).argsort()][0]
                scores[ranking_type].append(top_score)
            elif ranking_type == "rsa_score":
                # The rsa score computer the conditional NLL of the source given the prediction
                # In this case, the lowest score is the best
                ranking_scores = df_slice[json_key_column].values
                top_score = slice_scores[ranking_scores.argsort()][0]
                scores[ranking_type].append(top_score)
    return {k: np.mean(v) for k, v in scores.items()}


def compute_frequencies(
    df, score_type, num_seq_per_source, ranking_types
) -> dict[str, float]:
    frequencies = {ranking_type: [] for ranking_type in ranking_types}
    for i in range(0, len(df), num_seq_per_source):
        df_slice = df.loc[i : i + num_seq_per_source - 1]
        for ranking_type, json_key_column in ranking_types.items():
            slice_scores = df_slice[score_type].values
            max_score = slice_scores.max()
            if ranking_type == "random":
                frequencies[ranking_type].append(
                    np.random.choice(slice_scores, size=1)[0] == max_score
                )
            elif ranking_type == "max":
                frequencies[ranking_type].append(slice_scores.max() == max_score)
            elif ranking_type in "direct_score":
                # The direct scores are the accumlated logits e.g. T5 model
                # The highest score is the best according to this decoding method
                ranking_scores = df_slice[json_key_column].values
                top_score = slice_scores[(-ranking_scores).argsort()][0]
                frequencies[ranking_type].append(top_score == max_score)
            elif ranking_type == "rsa_score":
                # The rsa score computer the conditional NLL of the source given the prediction
                # In this case, the lowest score is the best
                ranking_scores = df_slice[json_key_column].values
                top_score = slice_scores[ranking_scores.argsort()][0]
                frequencies[ranking_type].append(top_score == max_score)
    return {k: np.sum(v) for k, v in frequencies.items()}


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rerank", "conf"),
)
@main_decorator
def main(run_name: str, cfg: DictConfig):
    # Read jsonlines data
    jsonlines_data = []
    with jsonlines.open(cfg["rescored_dataset_path"]) as reader:
        for obj in reader:
            jsonlines_data.append(obj)
    # Convert to df
    df = pd.DataFrame(jsonlines_data)
    dir_path = os.path.join(cfg["output_results_directory"], run_name)
    os.makedirs(dir_path, exist_ok=True)
    for eval_name, eval_info in cfg["evaluation"].items():
        if eval_name == "reranked_scores":
            df_out = compute_reranked_scores(df=df, eval_info=eval_info)
            with open(os.path.join(dir_path, f"{eval_name}.txt"), "a") as f:
                f.write(tabulate(df_out, headers="keys", tablefmt="psql"))
        elif eval_name == "reranked_frequencies":
            df_out = compute_reranked_frequencies(df=df, eval_info=eval_info)
            with open(os.path.join(dir_path, f"{eval_name}.txt"), "a") as f:
                f.write(tabulate(df_out, headers="keys", tablefmt="psql"))


if __name__ == "__main__":
    main()
