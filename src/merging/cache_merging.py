import os

import hydra
import pickle as pkl
import pandas as pd

from src import SRC_DIRECTORY
from src.rescoring.merge_rescorings import rename_columns
from src.rescoring.rescore_summaries import get_latent_column_name, process_llama2_preds
from src.utils.dataset import get_question_column_name
from src.utils.decorators import main_decorator


def get_printer(print_sanity_check=False):
    if print_sanity_check:
        return print
    else:
        return lambda _: None


def get_scores(df, cache, cache_key, col1, col2):
    return [
        cache.get((x1, x2), {cache_key: None})[cache_key]
        for x1, x2 in zip(df[col1], df[col2])
    ]


def add_scores(dataset, dataset_name, reconstruction_type, cached_rescorings):
    if reconstruction_type == "source_reconstruction":
        dataset["reconstruction_score"] = get_scores(
            df=dataset,
            cache=cached_rescorings,
            cache_key="reconstruction_score",
            col1="pred",
            col2="document",
        )
        dataset["avg_reconstruction_score"] = get_scores(
            df=dataset,
            cache=cached_rescorings,
            cache_key="avg_reconstruction_score",
            col1="pred",
            col2="document",
        )
    elif reconstruction_type == "latent_reconstruction":
        latent_column_name = get_latent_column_name(dataset_name)
        dataset["reconstruction_score"] = get_scores(
            df=dataset,
            cache=cached_rescorings,
            cache_key="reconstruction_score",
            col1="pred",
            col2=latent_column_name,
        )
        dataset["avg_reconstruction_score"] = get_scores(
            df=dataset,
            cache=cached_rescorings,
            cache_key="avg_reconstruction_score",
            col1="pred",
            col2=latent_column_name,
        )
    else:
        raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")
    dataset = rename_columns(dataset, reconstruction_type)
    return dataset


def get_preds(dataset, cache, col1, col2):
    dataset_iteration = (
        [(elt,) for elt in dataset[col1]]
        if col2 is None
        else [(x1, x2) for x1, x2 in zip(dataset[col1], dataset[col2])]
    )
    dataset["pred"] = [
        [pred_dict["pred"] for pred_dict in cache.get(elt, [{"pred": None}])]
        for elt in dataset_iteration
    ]
    dataset["pred_score"] = [
        [
            pred_dict["pred_score"]
            for pred_dict in cache.get(elt, [{"pred_score": None}])
        ]
        for elt in dataset_iteration
    ]
    return dataset


def assign_preds(dataset, dataset_name, summarizer_name, summaries_cache):
    if "generic" in summarizer_name:
        dataset = get_preds(
            dataset=dataset,
            cache=summaries_cache,
            col1="document",
            col2=None,
        )
    elif "e2e" in summarizer_name:
        question_column_name = get_question_column_name(dataset_name)
        if dataset_name == "covidet":
            dataset[question_column_name] = convert_emotion_to_question(dataset)
        dataset = get_preds(
            dataset=dataset,
            cache=summaries_cache,
            col1="document",
            col2=question_column_name,
        )
    else:
        raise ValueError(f"Unknown summarizer name: {summarizer_name}")
    return dataset

def convert_emotion_to_question(dataset):
    return dataset["emotion"].apply(
                lambda emotion: f"Describe the {emotion} of this post."
            )


def merge(
    datasets,
    summarizer_names,
    cached_summaries,
    cached_rescorings,
    print_sanity_check,
):
    print_sanity_check = get_printer(print_sanity_check)
    datasets_dict = {}
    for summarizer_name in summarizer_names:
        for dataset_name, dataset_path in datasets.items():
            print_sanity_check("=" * 80)
            print_sanity_check(
                f"Summarizer: {summarizer_name} - Dataset: {dataset_name}"
            )
            dataset = pd.read_csv(dataset_path)
            # Assign predictions and prediction scores to dataset
            dataset = assign_preds(
                dataset=dataset,
                dataset_name=dataset_name,
                summarizer_name=summarizer_name,
                summaries_cache=cached_summaries[summarizer_name],
            )
            print_sanity_check(
                f"Documents missing at least one pred: [{dataset['pred'].apply(lambda x: x[0]).isna().sum()}]"
            )
            # Explode the dataset
            dataset_exploded = dataset.explode(["pred", "pred_score"])
            if "llama" in summarizer_name:
                dataset_exploded["pred"] = process_llama2_preds(
                    dataset_exploded["pred"].values.tolist()
                )
            print_sanity_check(
                f"Number of initial rows : {len(dataset)}, number of expected exploded rows: {len(dataset) * 5}, number of exploded rows: {len(dataset_exploded)}"
            )
            # Add source reconstruction scores
            dataset_exploded = add_scores(
                dataset=dataset_exploded,
                dataset_name=dataset_name,
                reconstruction_type="source_reconstruction",
                cached_rescorings=cached_rescorings,
            )
            print_sanity_check(
                f"Number of missing source rescoring: {sum(dataset_exploded['source_rec_score'].isnull())}"
            )
            print_sanity_check(
                f"Number of missing source rescoring: {sum(dataset_exploded['source_avg_rec_score'].isnull())}"
            )
            # Add latent reconstruction scores
            dataset_exploded = add_scores(
                dataset=dataset_exploded,
                dataset_name=dataset_name,
                reconstruction_type="latent_reconstruction",
                cached_rescorings=cached_rescorings,
            )
            print_sanity_check(
                f"Number of missing latent rescoring: {sum(dataset_exploded['latent_rec_score'].isnull())}"
            )
            print_sanity_check(
                f"Number of missing latent rescoring: {sum(dataset_exploded['latent_avg_rec_score'].isnull())}"
            )
            # Write merged dataset
            datasets_dict[(summarizer_name, dataset_name)] = dataset_exploded
    return datasets_dict


def write_merged_datasets(datasets_dict, output_dir_path, run_name, convert_new_line):
    for (summarizer_name, dataset_name), dataset in datasets_dict.items():
        path_to_write = os.path.join(output_dir_path, run_name, summarizer_name)
        if not os.path.exists(path_to_write):
            os.makedirs(path_to_write)
        output_path = os.path.join(
            path_to_write,
            f"{dataset_name}.csv",
        )
        if convert_new_line:
            for col in dataset.columns:
                dataset[col] = dataset[col].apply(
                    lambda x: x.replace("\n", "\\n") if isinstance(x, str) else x
                )
        dataset.to_csv(output_path, index=False)


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "merging", "conf", "cache_merging"),
    config_name="config",
)
@main_decorator
def main(run_name, cfg):
    # Get cached summaries
    cached_summaries = {}
    for summarizer_name in cfg["summarizer_names"]:
        with open(
            os.path.join(cfg["cached_summaries_dir_path"], f"{summarizer_name}.pkl"),
            "rb",
        ) as f:
            cached_summaries[summarizer_name] = pkl.load(f)
    # Get cached rescores
    cached_rescorings = {}
    with open(cfg["cached_rescores_path"], "rb") as f:
        cached_rescorings = pkl.load(f)
    datasets_dict = merge(
        datasets=cfg["datasets"],
        summarizer_names=cfg["summarizer_names"],
        cached_summaries=cached_summaries,
        cached_rescorings=cached_rescorings,
        print_sanity_check=cfg["print_sanity_check"],
    )
    write_merged_datasets(
        datasets_dict, cfg["output_directory"], run_name, cfg["convert_new_line"]
    )


if __name__ == "__main__":
    main()
