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
        negative_label,
    ):

    labels = {}
    with open(label_filepath) as f:
        freader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for line in freader:
            video, label = line[0], line[1]
            labels[video] = label

    if num_negatives_set:
        negative_labels = {}
        negative_test_labels = {}
        train_val_labels = {}
        negatives_sampled = 0
        negatives_test_sampled = 0
        for k, v in labels.items():
            if (v == str(negative_label)) & (negatives_sampled < num_negatives_set):
                negative_labels[k] = v
                negatives_sampled += 1
            elif (v == str(negative_label)) & (negatives_test_sampled < num_negatives_test):
                negative_test_labels[k] = v
                negatives_test_sampled += 1
            else:
                train_val_labels[k] = v

        negatives_samples = random.sample(negative_labels.items(), k=len(negative_labels))
        output_split(negatives_samples, "neg_set.txt")
        negatives_test_samples = random.sample(negative_test_labels.items(), k=len(negative_test_labels))
        output_split(negatives_test_samples, "neg_test.txt")

    samples = random.sample(train_val_labels.items(), k=len(train_val_labels))
    split_point = math.floor(train_proportion*len(samples))
    train = samples[:split_point]
    val = samples[(split_point+1):]

    output_split(train, "train.txt")
    output_split(val, "val.txt")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-L",
        "--label_filepath",
        help="Path where the label csv will be stored",
        default="./outputs/labels.csv",
    )
    parser.add_argument(
        "-T",
        "--train_proportion",
        help="Proportion of examples to go in training set",
        default=0.75
    )
    args = parser.parse_args()

    main(
        label_filepath="labels.txt",
        train_proportion=0.75,
        num_negatives_set=400,
        num_negatives_test=50,
        negative_label=1
    )

