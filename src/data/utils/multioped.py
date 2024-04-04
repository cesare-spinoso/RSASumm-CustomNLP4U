"""
Code written and taken from: https://raw.githubusercontent.com/CogComp/MultiOpEd/main/utils.py
"""

import argparse
import csv
import random
import torch
import numpy as np
import os


def load_csv(path):
    """loading the csv data file"""

    source = []
    target = []
    query = []
    with open(path, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # print(row)
            query.append(row[0])
            source.append(row[4])
            target.append(row[3])

    source = source[1:]
    target = target[1:]
    query = query[1:]

    print(len(source))
    print(len(target))
    print(len(query))

    for i in range(len(source)):
        source[i] = source[i].replace("\n", "")
        target[i] = target[i].replace("\n", "")
        query[i] = query[i].replace("\n", "")

    # print(len(total_texts))
    # randomize the train/dev/test/ split
    total_texts = [
        (source[i], source[i + 1], query[i]) for i in range(0, len(source) - 1, 2)
    ]
    total_labels = [
        (target[i], target[i + 1], query[i]) for i in range(0, len(target) - 1, 2)
    ]
    # print(total_texts[:3])
    # print(total_labels[:3])

    # print(total_query[:3])
    random.Random(4).shuffle(total_texts)
    random.Random(4).shuffle(total_labels)
    # random.Random(4).shuffle(total_query)
    print(total_texts[:3])
    print(total_labels[:3])
    print(len(total_texts))
    # print(total_query[:3])

    train_len = len(total_texts) * 7 // 10
    dev_len = len(total_texts) * 8 // 10
    print(train_len)

    train_texts = []
    train_labels = []
    train_query = []

    dev_texts = []
    dev_labels = []
    dev_query = []

    test_texts = []
    test_labels = []
    test_query = []

    for i in range(train_len):
        train_texts.append(total_texts[i][0])
        train_texts.append(total_texts[i][1])
        train_labels.append(total_labels[i][0])
        train_labels.append(total_labels[i][1])
        train_query.append(total_texts[i][2])
        train_query.append(total_labels[i][2])

    for i in range(train_len, dev_len):
        dev_texts.append(total_texts[i][0])
        dev_texts.append(total_texts[i][1])
        dev_labels.append(total_labels[i][0])
        dev_labels.append(total_labels[i][1])
        dev_query.append(total_texts[i][2])
        dev_query.append(total_labels[i][2])

    for i in range(dev_len, len(total_texts)):
        test_texts.append(total_texts[i][0])
        test_texts.append(total_texts[i][1])
        test_labels.append(total_labels[i][0])
        test_labels.append(total_labels[i][1])
        test_query.append(total_texts[i][2])
        test_query.append(total_labels[i][2])

    dic = {}
    for i in range(len(train_labels)):
        dic[train_labels[i]] = train_query[i]
        if train_query[i] == "was trump right to kill soleimani?":
            print("here", train_labels[i])

    for i in range(len(dev_labels)):
        dic[dev_labels[i]] = dev_query[i]

    for i in range(len(test_labels)):
        dic[test_labels[i]] = test_query[i]

    return dic


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--data_path", type=str)
    # args = parser.parse_args()
    # dic = load_csv(args.data_path)
    # print(dic)
    load_csv("/home/mila/c/cesare.spinoso/RSASumm/data/multioped/raw.csv")