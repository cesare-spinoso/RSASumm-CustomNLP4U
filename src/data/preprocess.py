# TODO:
# 1. For debatepedia, merege all the files
# 2. For all the files, flatten the structure (if not flattened yet)
# and save under the same column-name schema
# 3. Add document ids

import json, jsonlines, os
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


def extract_data(raw_directory, files):
    data = []
    for file_name in files:
        with open(os.path.join(raw_directory, file_name), "r") as f:
            data.extend(f.readlines())
    # remove <s>, <eos>
    data = [d.replace("<s>", "").replace("<eos>", "").strip() for d in data]
    return data


def debatepedia_preprocess(cfg):
    # Read and merge files of the same type together
    raw_directory = cfg["raw_data_dir_path"]
    content_data = extract_data(raw_directory, files=cfg["content_files"])
    summary_data = extract_data(raw_directory, files=cfg["summary_files"])
    query_data = extract_data(raw_directory, files=cfg["query_files"])
    assert len(content_data) == len(summary_data) == len(query_data)
    # For query, split by : left is topic, right is query
    indices = []
    topics = []
    queries = []
    for i, q in enumerate(query_data):
        split_q = q.split(":")
        if len(split_q) == 2:
            indices.append(i)
            topics.append(split_q[0])
            queries.append(split_q[1])
    content_data = [content_data[i] for i in indices]
    summary_data = [summary_data[i] for i in indices]
    assert len(content_data) == len(summary_data) == len(queries) == len(topics)
    data = pd.DataFrame(
        {
            "document": content_data,
            "summary": summary_data,
            "query": queries,
            "topic": topics,
        }
    )
    column_names = data.columns.tolist()
    for col in cfg["column_names"]:
        if col not in column_names:
            data[col] = None
    return data


def extract_qmsum_doc(document_list: list[dict], relevant_text_span: list[list]) -> str:
    relevant_text_span = [[int(elt) for elt in outer] for outer in relevant_text_span]
    selected_lines = []
    for outer in relevant_text_span:
        # Page 4 of their paper : It looks like its inclusive
        selected_lines += document_list[outer[0] : outer[1] + 1]
    merged_speaker_utterance = [
        elt["speaker"] + " : " + elt["content"] for elt in selected_lines
    ]
    return " ".join(merged_speaker_utterance)


def qmsum_preprocess(cfg):
    raw_directory = cfg["raw_data_dir_path"]
    data = []
    for file_name in cfg["files"]:
        file_path = os.path.join(raw_directory, file_name)
        with jsonlines.open(file_path) as reader:
            for obj in reader:
                data.append(obj)
    preprocessed_data = dict.fromkeys(cfg["column_names"], None)
    preprocessed_data = {k: [] for k in preprocessed_data.keys()}
    for json_dict in data:
        document_list = json_dict["meeting_transcripts"]
        for query_dict in json_dict["specific_query_list"]:
            preprocessed_data["question"] += query_dict["query"]
            preprocessed_data["summary"] += query_dict["answer"]
            preprocessed_data["document"] += extract_qmsum_doc(
                document_list, query_dict["relevant_text_span"]
            )
    return preprocessed_data


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
    elif cfg["dataset_name"] == "qmsum":
        preprocessed_data = qmsum_preprocess(cfg)
    elif cfg["dataset_name"] == "tac":
        raise ValueError("Not yet implemented!")
    elif cfg["dataset_name"] == "duc":
        raise ValueError("Not yet implemented!")
    else:
        raise ValueError(f"No preprocessing for {cfg['dataset_name']} dataset")
    pass
    write_preprocesed_data(preprocessed_data, cfg)


if __name__ == "__main__":
    main()
