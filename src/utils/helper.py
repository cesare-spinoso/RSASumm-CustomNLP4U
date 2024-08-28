import os
from itertools import product

import pandas as pd
import json
import jsonlines
import yaml
from tqdm import tqdm

# JSON/JSONLINES Helpers


def read_jsonlines(file_path):
    jsonlines_dict = []
    with jsonlines.open(file_path, "r") as reader:
        for elt in reader:
            jsonlines_dict.append(elt)
    return jsonlines_dict


def get_values_from_jsonlines(jsonlines_data: list[dict], keys: list[str]) -> tuple:
    values = {}
    for key in keys:
        values[key] = [elt[key] for elt in jsonlines_data]
    return (values[key] for key in keys)


def append_jsonlines(dict_to_write, output_directory, run_name):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with jsonlines.open(
        os.path.join(output_directory, f"{run_name}.jsonl"), "a"
    ) as writer:
        writer.write_all(dict_to_write)


def group_jsonlines(grouping_keys, jsonlines_data):
    grouping_key_dict = {}
    for elt in jsonlines_data:
        candidate_tuple = tuple([elt[k] for k in grouping_keys])
        if candidate_tuple not in grouping_key_dict.keys():
            grouping_key_dict[candidate_tuple] = [elt]
        else:
            grouping_key_dict[candidate_tuple].append(elt)
    return grouping_key_dict


def filter_jsonlines(keys, values, jsonlines_data, remove_keys=False):
    assert len(keys) == len(values)
    filtered_jsonlines_data = []
    for elt in jsonlines_data:
        if all(elt[k] == v for k, v in zip(keys, values)):
            if not remove_keys:
                filtered_jsonlines_data.append(elt)
            else:
                filtered_jsonlines_data.append(
                    {k: v for k, v in elt.items() if k not in keys}
                )
    return filtered_jsonlines_data


def rename_jsonlines_keys(original_keys, new_keys, jsonlines_data):
    assert len(original_keys) == len(new_keys)
    new_jsonlines_data = []
    for elt in jsonlines_data:
        new_elt = {k: v for k, v in elt.items() if k not in original_keys}
        new_elt = {
            **new_elt,
            **{k_new: elt[k_og] for k_og, k_new in zip(original_keys, new_keys)},
        }
        new_jsonlines_data.append(new_elt)
    return new_jsonlines_data


def merge_jsonlines_data(jsonlines_paths):
    jsonlines_data = []
    for jsonlines_path in jsonlines_paths:
        jsonlines_data.extend(read_jsonlines(jsonlines_path))
    merged_jsonlines_data = []
    for elt in jsonlines_data:
        if elt not in merged_jsonlines_data:
            merged_jsonlines_data.append(elt)
    return merged_jsonlines_data


def write_json_file(dict_to_write, output_directory, run_name):
    with open(os.path.join(output_directory, f"{run_name}.json"), "w") as f:
        json.dump(dict_to_write, f, indent=4)


def read_json_file(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


# YAML Helpers


def read_yaml(yaml_file_path):
    with open(yaml_file_path, "r") as f:
        yaml_contents = yaml.safe_load(f)
    return yaml_contents


def key_index_dict(ordered_keys, dictionary):
    # Cool use of recursion!
    if len(ordered_keys) == 1:
        return dictionary[ordered_keys[0]]
    else:
        return key_index_dict(ordered_keys[1:], dictionary[ordered_keys[0]])


def get_jsonl_path_from_yaml(ordered_keys, yaml_path):
    with open(yaml_path, "r") as yaml_file:
        try:
            yaml_contents = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return key_index_dict(ordered_keys, yaml_contents)


def yaml_to_df(yaml_file_path):
    yaml_contents = read_yaml(yaml_file_path)
    data = {}
    for k, vdict in yaml_contents.items():
        data[k] = ["/".join(["" if v is None else f"{v:.3f}" for v in vdict.values()])]
    return pd.DataFrame(data)
