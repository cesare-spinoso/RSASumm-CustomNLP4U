import os
import pickle as pkl

import hydra
from tqdm import tqdm

from src import SRC_DIRECTORY
from src.rescoring.rescore_summaries import get_latent_column_name
from src.utils.decorators import main_decorator
from src.utils.helper import read_jsonlines, read_yaml


def get_cache_content_from_jsonlines(
    cache_input_keys, cache_output_keys, jsonlines_data
):
    cache_content = {}
    for elt in jsonlines_data:
        key = tuple(elt[k] for k in cache_input_keys)
        value = {k: elt[k] for k in cache_output_keys}
        cache_content[key] = value
    return cache_content


def get_cache_content_from_yaml(
    yaml_content, rec_type, summarizer_names=None, dataset_names=None
):
    cache_content = {}
    cache_output_keys = ["reconstruction_score", "avg_reconstruction_score"]
    for summarizer_name, jsonl_path_dict in tqdm(yaml_content.items()):
        if (
            isinstance(summarizer_names, list)
            and summarizer_name not in summarizer_names
        ):
            continue
        for dataset_name, jsonlines_path in tqdm(jsonl_path_dict.items()):
            if isinstance(dataset_name, list) and dataset_name not in dataset_names:
                continue
            latent_column_name = get_latent_column_name(dataset_name)
            if rec_type == "source_reconstruction":
                cache_input_keys = ["pred", "source"]
            elif rec_type == "latent_reconstruction":
                cache_input_keys = ["pred", latent_column_name]
            else:
                raise ValueError(f"Unsupported rec_type: {rec_type}")
            jsonlines_data = read_jsonlines(jsonlines_path)
            cache_content = {
                **cache_content,
                **get_cache_content_from_jsonlines(
                    cache_input_keys, cache_output_keys, jsonlines_data
                ),
            }
    return cache_content


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "rescoring", "conf", "populate_cache"),
    config_name="config",
)
@main_decorator
def main(_, config):
    # Load the cache
    print("Loading the cache")
    cache_path = os.path.join(
        config["cache_directory"], f"{config['reconstruction_model_name']}.pkl"
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache_dict = pkl.load(f)
    else:
        cache_dict = {}
    # Cache rescorings from jsonlines found in yamls
    print("Fetching rescorings from jsonlines")
    yaml_paths = config["reconstruction_yaml_paths"]
    summarizer_names = config.get("summarizer_names", None)
    dataset_names = config.get("dataset_names", None)
    for rec_type, yaml_path in yaml_paths.items():
        yaml_content = read_yaml(yaml_path)
        cache_dict = {
            **cache_dict,
            **get_cache_content_from_yaml(
                yaml_content,
                rec_type,
                summarizer_names=summarizer_names,
                dataset_names=dataset_names,
            ),
        }
    # Write to the cache
    print("Writing to the cache")
    with open(cache_path, "wb") as f:
        pkl.dump(cache_dict, f)


if __name__ == "__main__":
    main()
