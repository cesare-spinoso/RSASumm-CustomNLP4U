import os
import pickle as pkl
import warnings

import hydra

from src import SRC_DIRECTORY
from src.utils.decorators import main_decorator
from src.utils.helper import read_jsonlines, read_yaml


def cache_jsonlines(cache_input_keys, cache_output_keys, jsonlines_data, cache_path):
    cache_content = {}
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache_content.update(pkl.load(f))
    for elt in jsonlines_data:
        key = tuple(elt[k] for k in cache_input_keys)
        value = {k: elt[k] for k in cache_output_keys}
        if key not in cache_content:
            cache_content[key] = [value]
        else:
            cache_content[key].append(value)
    with open(cache_path, "wb") as f:
        pkl.dump(cache_content, f)


def cache_summaries_from_yaml(yaml_content, cache_dir):
    for model_name, jsonl_path_dict in yaml_content.items():
        cache_input_keys = ["source"]
        cache_output_keys = ["pred", "pred_score"]
        if "e2e" in model_name:
            cache_input_keys.append("question")
        for jsonlines_path in jsonl_path_dict.values():
            jsonlines_data = read_jsonlines(jsonlines_path)
            cache_path = os.path.join(cache_dir, f"{model_name}.pkl")
            warnings.warn(
                f"Note that the cache file is saved at {cache_path} which may be undesirable if you are dealing"
                + " with an updated model e.g. after more hyperparameter tuning."
            )
            cache_jsonlines(
                cache_input_keys, cache_output_keys, jsonlines_data, cache_path
            )


@hydra.main(
    version_base=None,
    config_path=os.path.join(
        SRC_DIRECTORY, "generate_summaries", "conf", "populate_cache"
    ),
)
@main_decorator
def main(_, config):
    yaml_paths = config["generate_summaries_yaml_paths"]
    for yaml_path in yaml_paths:
        yaml_content = read_yaml(yaml_path)
        cache_summaries_from_yaml(yaml_content, config["cache_directory"])


if __name__ == "__main__":
    main()
