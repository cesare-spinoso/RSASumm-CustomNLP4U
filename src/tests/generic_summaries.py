import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src import DATASET_NAMES, SRC_DIRECTORY
from src.utils.helper import get_jsonl_path_from_yaml, read_jsonlines

GENERIC_SUMMARIZERS = ["bart_generic", "t5_generic", "peg_generic", "led_generic"]


@hydra.main(version_base=None, config_path=os.path.join(SRC_DIRECTORY, "tests", "conf"))
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    print("=" * 80)
    print("Computing lengths in number of lines for each dataset and summarizer")
    print("Generate summaries")
    generations_lengths = verify_lengths(cfg["generated_summaries_yaml"])
    print(generations_lengths)
    print("Source reconstruction")
    generations_lengths = verify_lengths(cfg["source_reconstruction_yaml"])
    print(generations_lengths)
    print("Latent reconstruction")
    generations_lengths = verify_lengths(cfg["latent_reconstruction_yaml"])
    print(generations_lengths)
    print("=" * 80)
    print("Compute empty predictions")
    empty_predictions = compute_empty_predictions(cfg["generated_summaries_yaml"])
    print(empty_predictions)


def compute_empty_predictions(yaml_path):
    empty_predictions = {}
    for ds_name in DATASET_NAMES:
        empty_predictions[ds_name] = {}
        for summarizer_name in GENERIC_SUMMARIZERS:
            jsonl_path = get_jsonl_path_from_yaml(
                ordered_keys=[summarizer_name, ds_name],
                yaml_path=yaml_path,
            )
            jsonlines_data = read_jsonlines(jsonl_path)
            empty_predictions[ds_name][summarizer_name] = len(
                [elt for elt in jsonlines_data if elt["pred"] == ""]
            )
    return empty_predictions


def verify_lengths(yaml_path):
    generations_lengths = {}
    for ds_name in DATASET_NAMES:
        generations_lengths[ds_name] = {}
        for summarizer_name in GENERIC_SUMMARIZERS:
            jsonl_path = get_jsonl_path_from_yaml(
                ordered_keys=[summarizer_name, ds_name],
                yaml_path=yaml_path,
            )
            jsonlines_data = read_jsonlines(jsonl_path)
            generations_lengths[ds_name][summarizer_name] = len(jsonlines_data)
    return generations_lengths


if __name__ == "__main__":
    main()
