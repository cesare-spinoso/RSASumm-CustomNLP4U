# TODO:
# 1. For debatepedia, merege all the files
# 2. For all the files, flatten the structure (if not flattened yet)
# and save under the same column-name schema
import json, os
import hydra
import pandas as pd
from src import SRC_DIRECTORY

from src.utils.decorators import main_decorator


def covidet_preprocess(cfg):
    # Read the json file
    with open(cfg["raw_file_path"], "r") as f:
        data = json.load(f)
    preprocessed_data = dict.fromkeys(cfg["column_names"], None)
    preprocessed_data = {k: [] for k in preprocessed_data.keys()}
    for _, post_content in data.items():
        document = post_content["Reddit Post"]
        annotations = post_content["Annotations"]
        for v in annotations.values():
            for annotations_dict in v:
                if (
                    len(annotations_dict) == 2
                    and "Emotion" in annotations_dict
                    and "Abstractive" in annotations_dict
                ):
                    emotion = annotations_dict["Emotion"]
                    summary = annotations_dict["Abstractive"]
        preprocessed_data["document"].append(document)
        preprocessed_data["emotion"].append(emotion)
        preprocessed_data["summary"].append(summary)
    assert (
        (num_samples := len(preprocessed_data["document"]))
        == len(preprocessed_data["emotion"])
        == len(preprocessed_data["summary"])
    )
    preprocessed_data = {
        k: v if len(v) > 0 else [None] * num_samples
        for k, v in preprocessed_data.items()
    }
    return preprocessed_data


def multioped_preprocess(cfg):
    # Read the csv file
    data = pd.read_csv(cfg["raw_file_path"])
    # Rename
    data.rename(columns=cfg["column_mapping"], inplace=True)
    # Add
    current_colnames = data.columns.tolist()
    for col in cfg["column_names"]:
        if col not in current_colnames:
            data[col] = None
    # Remove
    cols_to_drop = [col for col in current_colnames if col not in cfg["column_names"]]
    data.drop(columns=cols_to_drop, inplace=True)
    return data


def debatepedia_preprocess(cfg):
    from src.data.debatepedia import DebatepediaPreprocessor

    dp = DebatepediaPreprocessor(cfg)
    dp.run()




def write_preprocesed_data(preprocessed_data, cfg):
    path_to_write = os.path.join(cfg["output_directory"], "preprocessed.csv")
    preprocessed_df = pd.DataFrame(preprocessed_data)
    preprocessed_df.to_csv(path_to_write, index=False)


@hydra.main(
    version_base=None,
    config_path=os.path.join(SRC_DIRECTORY, "data", "conf"),
)
@main_decorator
def main(run_name: str, cfg: dict):
    if cfg["dataset_name"] == "debatepedia":
        preprocessed_data = debatepedia_preprocess(cfg)
    elif cfg["dataset_name"] == "multioped":
        preprocessed_data = multioped_preprocess(cfg)
    elif cfg["dataset_name"] == "covidet":
        preprocessed_data = covidet_preprocess(cfg)
    else:
        raise ValueError(f"No preprocessing for {cfg['dataset_name']} dataset")
    pass
    write_preprocesed_data(preprocessed_data, cfg)


if __name__ == "__main__":
    main()
