import jsonlines
import os

from itertools import product
from tqdm import tqdm
import yaml

# JSON/JSONLINES Helpers


def read_jsonlines(file_path):
    jsonlines_dict = []
    with jsonlines.open(file_path, "r") as reader:
        for elt in reader:
            jsonlines_dict.append(elt)
    return jsonlines_dict


def append_jsonlines(dict_to_write, output_directory, run_name):
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


def merge_jsonlines_data(common_keys, jsonlines_data):
    # Filter based on common keys
    common_keys_data = []
    for jsonlines_file in jsonlines_data:
        data = []
        for elt in jsonlines_file:
            data.append(tuple([elt[k] for k in common_keys]))
        data = set(data)
        common_keys_data.append(data)
    intersect_keys_data = set.intersection(*common_keys_data)
    print(
        f"You had {len(intersect_keys_data)} common keys"
        + f" when the max was {max([len(k) for k in common_keys_data])}"
    )
    merged_jsonlines_data = []
    for key_data in tqdm(intersect_keys_data):
        starting_dict = dict(zip(common_keys, key_data))
        possible_combinations = product(
            *[
                filter_jsonlines(
                    keys=common_keys,
                    values=key_data,
                    jsonlines_data=j,
                    remove_keys=True,
                )
                for j in jsonlines_data
            ]
        )
        for elt in possible_combinations:
            merged_jsonlines_data.append(
                {
                    **starting_dict,
                    **{k: v for sub_dict in elt for k, v in sub_dict.items()},
                }
            )
    return merged_jsonlines_data


# YAML Helpers


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
