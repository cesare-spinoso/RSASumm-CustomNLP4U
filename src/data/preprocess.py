import json
import os
from xml.etree import ElementTree as ET

import hydra
import jsonlines
import pandas as pd

from src import SRC_DIRECTORY
from src.utils.decorators import main_decorator


def covidet_preprocess(cfg):
    # Read the json file
    # NOTE: There are some emotions that are repeated for the same document
    with open(cfg["raw_file_path"], "r") as f:
        data = json.load(f)
    preprocessed_data = dict.fromkeys(cfg["column_names"], None)
    preprocessed_data = {k: [] for k in preprocessed_data.keys()}
    for post_id, post_content in data.items():
        post_id = int(post_id)
        document = post_content["Reddit Post"]
        annotations = post_content["Annotations"]
        for v in annotations.values():
            for annotations_dict in v:
                if (
                    len(annotations_dict) == 2
                    and "Emotion" in annotations_dict
                    and "Abstractive" in annotations_dict
                ):
                    # Map
                    emotion = annotations_dict["Emotion"]
                    summary = annotations_dict["Abstractive"]
                    # Append
                    preprocessed_data["document_id"].append(post_id)
                    preprocessed_data["document"].append(document)
                    preprocessed_data["emotion"].append(emotion)
                    preprocessed_data["summary"].append(summary)
    assert (
        (num_samples := len(preprocessed_data["document"]))
        == len(preprocessed_data["document_id"])
        == len(preprocessed_data["emotion"])
        == len(preprocessed_data["summary"])
    )
    preprocessed_data = {
        k: v if len(v) > 0 else [None] * num_samples
        for k, v in preprocessed_data.items()
    }
    return preprocessed_data


def multioped_preprocess(cfg):
    # NOTE: This means that there might not necessarily be 2
    # opposing opinions for a given question
    # Read the csv file
    data = pd.read_csv(cfg["raw_file_path"])
    # Remove nans and rows which have non-sting values
    data = data.dropna()
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
    # Remove columns which don't have strings in column_mapping
    for col_to_check in cfg["column_mapping"].values():
        data = data[data[col_to_check].apply(lambda x: isinstance(x, str))]
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
            preprocessed_data["question"].append(query_dict["query"])
            preprocessed_data["summary"].append(query_dict["answer"])
            preprocessed_data["document"].append(
                extract_qmsum_doc(document_list, query_dict["relevant_text_span"])
            )
    assert (
        (num_samples := len(preprocessed_data["document"]))
        == len(preprocessed_data["question"])
        == len(preprocessed_data["summary"])
    )
    preprocessed_data = {
        k: v if len(v) > 0 else [None] * num_samples
        for k, v in preprocessed_data.items()
    }
    return preprocessed_data


def get_duc_document_strings(source_files_dir_path, topic_code):
    document_strings = []
    for filename in os.listdir(os.path.join(source_files_dir_path, topic_code)):
        file_path = os.path.join(source_files_dir_path, topic_code, filename)
        with open(file_path, "r") as f:
            content = f.read()
            start = content.find("<TEXT>")
            end = content.find("</TEXT>")
            end += len("</TEXT>")
            content = content[start:end]
            content = content.replace("<TEXT>", "").replace("</TEXT>", "")
            content = content.replace("<P>", "").replace("</P>", "")
        document_strings.append(content)
    assert len(document_strings) == 25
    return document_strings


def get_reference_summaries(reference_summaries_dir_path, topic_code):
    reference_summaries = []
    for filename in os.listdir(reference_summaries_dir_path):
        filename_split = filename.split(".")
        if (filename_split[0] + filename_split[3]) == topic_code and filename_split[
            4
        ].isalpha():
            file_path = os.path.join(reference_summaries_dir_path, filename)
            with open(file_path, "r", encoding="latin-1") as f:
                reference_summaries.append(f.read())
    assert len(reference_summaries) == 4
    return reference_summaries


def populate_data(data, topic, question, document_strings, reference_summaries):
    for document_string in document_strings:
        for reference_summary in reference_summaries:
            data["topic"].append(topic)
            data["question"].append(question)
            data["document"].append(document_string)
            data["summary"].append(reference_summary)
    return data


def duc_preprocess(cfg):
    # NOTE: This will require a max over reference summary rouge scores
    # Only support single-doc summarization
    assert cfg["n_source_docs"] == 1 and cfg["n_ref_summs"] == 1
    # Extract topic and corresponding document id
    topics_file_path = cfg["topics_file_path"]
    tree = ET.parse(topics_file_path)
    topic_codes = {}
    for elt in tree.getroot():
        assert elt.tag == "topic"
        topic_code = elt.find("num").text.strip()
        topic = elt.find("title").text.strip()
        question = elt.find("narr").text.replace("\n", " ").strip()
        topic_codes[topic_code] = {"topic": topic, "question": question}
    # Extract documents corresponding to every topic
    source_files_dir_path = cfg["source_files_dir_path"]
    reference_summaries_dir_path = cfg["manual_annotation_dir_path"]
    preprocessed_data = dict.fromkeys(cfg["column_names"], None)
    preprocessed_data = {k: [] for k in preprocessed_data.keys()}
    num_reference_summaries = 0
    num_documents = 0
    for topic_code in topic_codes:
        # Get document strings
        document_strings = get_duc_document_strings(source_files_dir_path, topic_code)
        num_documents += len(document_strings)
        # Get the reference summary strings
        reference_summaries = get_reference_summaries(
            reference_summaries_dir_path, topic_code
        )
        num_reference_summaries += len(reference_summaries)
        # Populate the preprocessed data
        preprocessed_data = populate_data(
            data=preprocessed_data,
            topic=topic_codes[topic_code]["topic"],
            question=topic_codes[topic_code]["question"],
            document_strings=document_strings,
            reference_summaries=reference_summaries,
        )
    assert num_reference_summaries == 4 * len(topic_codes)
    assert num_documents == 25 * len(topic_codes)
    assert (
        (num_samples := len(preprocessed_data["document"]))
        == len(preprocessed_data["question"])
        == len(preprocessed_data["topic"])
        == len(preprocessed_data["summary"])
    )
    preprocessed_data = {
        k: v if len(v) > 0 else [None] * num_samples
        for k, v in preprocessed_data.items()
    }
    return preprocessed_data


def add_document_ids(df):
    if "document_id" in df.columns and df["document_id"].isna().sum() == 0:
        return df
    elif "document_id" not in df.columns:
        df["document_id"] = None
    documents = df["document"].tolist()
    document_set = set(documents)
    document_ids = [None for _ in documents]
    document_count = 0
    for document in document_set:
        document_ids = [
            document_count if d == document else document_ids[i]
            for i, d in enumerate(documents)
        ]
        document_count += 1
    df["document_id"] = document_ids
    return df


def write_preprocesed_data(preprocessed_data, cfg):
    path_to_write = os.path.join(cfg["output_directory"], "preprocessed.csv")
    if not isinstance(preprocessed_data, pd.DataFrame):
        preprocessed_data = pd.DataFrame(preprocessed_data)
    # Sort the column names
    preprocessed_data = preprocessed_data.reindex(sorted(preprocessed_data.columns), axis=1)
    # Populate document ids if not done yet
    preprocessed_data = add_document_ids(preprocessed_data)
    preprocessed_data.to_csv(path_to_write, index=False)


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
    elif cfg["dataset_name"] == "duc_single":
        preprocessed_data = duc_preprocess(cfg)
    else:
        raise ValueError(f"No preprocessing for {cfg['dataset_name']} dataset")
    pass
    write_preprocesed_data(preprocessed_data, cfg)


if __name__ == "__main__":
    main()
