from src.utils.helper import read_jsonlines, read_yaml
import os
from pathlib import Path


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


yaml_path = "/home/mila/c/cesare.spinoso/RSASumm/src/generate_summaries/generated_summaries/qfs_summaries.yaml"

path = "/home/mila/c/cesare.spinoso/RSASumm/src/generate_summaries/config_instances"

expected_jsonl_size = {
    "multioped": 5310,
    "qmsum": 13650,
    "squality": 2520,
}

yaml_data = read_yaml(yaml_path)

for summarizer_name, dataset_to_yaml_paths in yaml_data.items():
    for dataset_name, dataset_jsonl_path in dataset_to_yaml_paths.items():
        if dataset_jsonl_path is None:
            continue
        jsonlines_data = read_jsonlines(dataset_jsonl_path)
        assert len(jsonlines_data) == expected_jsonl_size[dataset_name], f"summarizer_name: {summarizer_name}, dataset_name: {dataset_name}"
        dataset_yaml_name = Path(dataset_jsonl_path).stem + ".yaml"
        dataset_yaml_path = find(dataset_yaml_name, path)
        dataset_yaml_data = read_yaml(dataset_yaml_path)
        assert dataset_yaml_data["dataset"]["name"] == dataset_name
        if "beam" in summarizer_name:
            assert dataset_yaml_data["finished_running"]
            assert dataset_yaml_data["generation"]["generate_kwargs"]["num_beams"] == 10
            assert not dataset_yaml_data["generation"]["generate_kwargs"]["do_sample"]
            assert all(
                elt not in dataset_yaml_data["generation"]["generate_kwargs"]
                or dataset_yaml_data["generation"]["generate_kwargs"][elt] is None
                for elt in ["top_p", "top_k", "temperature"]
            )
        elif "topk" in summarizer_name:
            assert dataset_yaml_data["finished_running"]
            assert "top_k" in dataset_yaml_data["generation"]["generate_kwargs"]
            assert dataset_yaml_data["generation"]["generate_kwargs"]["do_sample"]
            assert all(
                elt not in dataset_yaml_data["generation"]["generate_kwargs"]
                or dataset_yaml_data["generation"]["generate_kwargs"][elt] is None
                for elt in ["num_beams", "top_p"]
            )
        elif "nucleus" in summarizer_name:
            assert dataset_yaml_data["finished_running"]
            assert "top_p" in dataset_yaml_data["generation"]["generate_kwargs"]
            assert dataset_yaml_data["generation"]["generate_kwargs"]["do_sample"]
            assert all(
                elt not in dataset_yaml_data["generation"]["generate_kwargs"]
                or dataset_yaml_data["generation"]["generate_kwargs"][elt] is None
                for elt in [
                    "num_beams",
                    "top_k",
                ]
            )
        else:
            assert dataset_yaml_data[
                "finished_running"
            ], f"summarizer_name: {summarizer_name}, dataset_name: {dataset_name}"
            assert dataset_yaml_data["generation"]["generate_kwargs"]["do_sample"]
            assert all(
                elt not in dataset_yaml_data["generation"]["generate_kwargs"]
                or dataset_yaml_data["generation"]["generate_kwargs"][elt] is None
                for elt in ["num_beams", "top_p", "top_k"]
            ), f"summarizer_name: {summarizer_name}, dataset_name: {dataset_name}"
        if "bart" in summarizer_name:
            path_to_checkpoint = dataset_yaml_data["model"]["path_to_checkpoint"]
            config_name = path_to_checkpoint.split("finetuning/")[-1].split("/lightning_logs")[0]
            finetuning_yaml = read_yaml(f"/home/mila/c/cesare.spinoso/RSASumm/src/finetuning/literal_summarizer/config_instances/bart/{config_name}.yaml")
            assert finetuning_yaml["data"]["name"] == dataset_name
