import os
import yaml

import hydra
import pandas as pd

from src import SRC_DIRECTORY, DATASET_NAMES
from src.utils.decorators import main_decorator
from src.utils.helper import append_jsonlines, read_jsonlines


def sanity_check_config(cfg):
    assert all(
        filtering_dict["summarizer_name"] in filtering_dict["summary_jsonl_dir"]
        for filtering_dict in cfg["to_filter"]
    )
    assert all(
        filtering_dict["dataset_name"] in filtering_dict["preprocessed_path"]
        and filtering_dict["dataset_name"] in DATASET_NAMES
        for filtering_dict in cfg["to_filter"]
    )
    for filtering_dict in cfg["to_filter"]:
        yaml_file_path = os.path.join(
            filtering_dict["summary_jsonl_dir"].replace(
                "data/generate_summaries", "src/generate_summaries/config_instances"
            ),
            filtering_dict["summary_jsonl_filename"].split("_")[0] + ".yaml",
        )
        with open(yaml_file_path, "r") as f:
            yaml_contents = yaml.safe_load(f)
        assert yaml_contents["dataset"]["name"] == filtering_dict["dataset_name"]
        assert filtering_dict["summarizer_name"] in yaml_contents["output_directory"]


def filter_summary(filtering_dict):
    summary_jsonlines = read_jsonlines(
        os.path.join(
            filtering_dict["summary_jsonl_dir"],
            filtering_dict["summary_jsonl_filename"],
        )
    )
    df_raw = pd.read_csv(filtering_dict["preprocessed_path"])
    sources = df_raw["document"].values.tolist()
    document_ids = df_raw["document_id"].values.tolist()
    filtered_summaries = []
    for elt in summary_jsonlines:
        if elt["source"] in sources and elt["document_id"] in document_ids:
            filtered_summaries.append(elt)
    print(
        f"Started with {len(summary_jsonlines)} rows and ended with {len(filtered_summaries)} rows."
    )
    print(f"This caused us to loose {len(summary_jsonlines) - len(filtered_summaries)} rows in the process :'(")
    append_jsonlines(
        filtered_summaries,
        filtering_dict["summary_jsonl_dir"],
        filtering_dict["summary_jsonl_filename"].replace(".jsonl","") + "_filtered",
    )


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "generate_summaries", "conf", "filter"),
    config_name="config",
)
@main_decorator
def main(_, cfg):
    sanity_check_config(cfg)
    for filtering_dict in cfg["to_filter"]:
        print(f"Filtering {filtering_dict} rescores.")
        filter_summary(filtering_dict)


if __name__ == "__main__":
    main()
