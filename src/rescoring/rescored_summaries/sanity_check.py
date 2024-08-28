from src.utils.helper import read_jsonlines, read_yaml
import os
from pathlib import Path


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


path = "/home/mila/c/cesare.spinoso/RSASumm/src/rescoring/config_instances"

expected_jsonl_size = {
    "multioped": 5310,
    "qmsum": 13650,
    "squality": 2520,
}

yaml_path = "/home/mila/c/cesare.spinoso/RSASumm/src/rescoring/rescored_summaries/answer_reconstruction.yaml"

yaml_data = read_yaml(yaml_path)

for summarizer_name, dataset_to_yaml_paths in yaml_data.items():
    for dataset_name, dataset_jsonl_path in dataset_to_yaml_paths.items():
        if dataset_jsonl_path is None or (
            dataset_name == "qmsum" and summarizer_name == "llama3_qfs_topk_640"
        ):
            continue
        jsonlines_data = read_jsonlines(dataset_jsonl_path)
        assert (
            len(jsonlines_data) == expected_jsonl_size[dataset_name]
        ), f"summarizer_name: {summarizer_name}, dataset_name: {dataset_name}"
        dataset_yaml_name = Path(dataset_jsonl_path).stem + ".yaml"
        dataset_yaml_path = find(dataset_yaml_name, path)
        dataset_yaml_data = read_yaml(dataset_yaml_path)
        assert dataset_name == dataset_yaml_data["dataset_name"]
        assert summarizer_name == dataset_yaml_data["summarizer_name"]
        assert dataset_yaml_data["finished_running"]
        assert dataset_yaml_data["rescoring"]["type"] == "answer_reconstruction"


yaml_path = "/home/mila/c/cesare.spinoso/RSASumm/src/rescoring/rescored_summaries/source_reconstruction.yaml"


yaml_data = read_yaml(yaml_path)

for summarizer_name, dataset_to_yaml_paths in yaml_data.items():
    for dataset_name, dataset_jsonl_path in dataset_to_yaml_paths.items():
        if dataset_jsonl_path is None or (
            dataset_name == "qmsum" and summarizer_name == "llama3_qfs_topk_640"
        ):
            continue
        jsonlines_data = read_jsonlines(dataset_jsonl_path)
        assert (
            len(jsonlines_data) == expected_jsonl_size[dataset_name]
        ), f"summarizer_name: {summarizer_name}, dataset_name: {dataset_name}, len(jsonlines_data): {len(jsonlines_data)}"
        dataset_yaml_name = Path(dataset_jsonl_path).stem + ".yaml"
        dataset_yaml_path = find(dataset_yaml_name, path)
        dataset_yaml_data = read_yaml(dataset_yaml_path)
        assert dataset_name == dataset_yaml_data["dataset_name"]
        assert summarizer_name == dataset_yaml_data["summarizer_name"]
        assert dataset_yaml_data["finished_running"]
        assert dataset_yaml_data["rescoring"]["type"] == "source_reconstruction"
