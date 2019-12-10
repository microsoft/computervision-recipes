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
    ):

    labels = {}
    with open(label_filepath) as f:
        freader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for line in freader:
            video, label = line[0], line[1]
            labels[video] = label

    samples = random.sample(labels.items(), k=len(labels))
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
    )

