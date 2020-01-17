# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Splits a video dataset created with clip_extraction.py into training and evaluation sets. Provide the output
label file in label_filepath and the proportion of examples to go to the training split with train_proportion.
Optionally, if hard negative mining is required, a specified number of negative examples can be reserved for
a negative candidates set or negative test set with the num_negatives_set and num_negatives_test parameters.
"""

import argparse
import random
import math
import csv

def output_split(labels_list, output_file):
    with open(output_file, 'a') as f:
        for label in labels_list:
            f.write("\""+label[0]+"\""+" "+label[1]+"\n")

def main(
        label_filepath,
        train_proportion,
        num_negatives_set,
        num_negatives_test,
        negative_label_id,
    ):

    labels = {}
    num_negative = 0
    num_non_negative = 0
    with open(label_filepath) as f:
        freader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for line in freader:
            video, label = line[0], line[1]
            labels[video] = label
            if negative_label_id:
                if label == negative_label_id:
                    num_negative += 1
                else:
                    num_non_negative += 1

    if num_negatives_set:
        if num_negatives_set + num_negatives_test > num_negative:
            raise Exception("Number of examples for negative candidate set and test set exceed number of negative examples")

        negative_label_ids = {}
        negative_test_labels = {}
        train_val_labels = {}
        negatives_sampled = 0
        negatives_test_sampled = 0
        for k, v in labels.items():
            if (v == str(negative_label_id)) & (negatives_sampled < num_negatives_set):
                negative_label_ids[k] = v
                negatives_sampled += 1
            elif (v == str(negative_label_id)) & (negatives_test_sampled < num_negatives_test):
                negative_test_labels[k] = v
                negatives_test_sampled += 1
            else:
                train_val_labels[k] = v

        negatives_samples = random.sample(negative_label_ids.items(), k=len(negative_label_ids))
        output_split(negatives_samples, "neg_set.txt")
        negatives_test_samples = random.sample(negative_test_labels.items(), k=len(negative_test_labels))
        output_split(negatives_test_samples, "neg_test.txt")
    else:
        train_val_labels = labels

    samples = random.sample(train_val_labels.items(), k=len(train_val_labels))
    split_point = math.floor(train_proportion*len(samples))
    train = samples[:split_point]
    val = samples[(split_point+1):]

    output_split(train, "train.txt")
    output_split(val, "val.txt")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_filepath",
        help="Path to the label file",
        required=True,
    )
    parser.add_argument(
        "--train_proportion",
        help="Proportion of examples to go in training set",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--num_negatives_set",
        help="Number of negative samples to include in a negative condidate set for hard negative mining",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--num_negatives_test",
        help="Number of negative samples to reserve for negative test set for hard negative mining",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--negative_label_id",
        help="Label of the negative class if applicable"
    )
    args = parser.parse_args()

    main(
        label_filepath=args.label_filepath,
        train_proportion=args.train_proportion,
        num_negatives_set=args.num_negatives_set,
        num_negatives_test=args.num_negatives_test,
        negative_label_id=args.negative_label_id
    )

