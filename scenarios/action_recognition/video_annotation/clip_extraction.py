# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# prerequisite:
# (1) download and extract ffmpeg: https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg
# (2) make sure the ffmpeg is in your system's env variable: path
# the script depend on the following fixed things of the csv
# skiprows=1

import argparse
import ast
import os
import sys

import pandas as pd

sys.path.append("../../../utils_cv/action_recognition/")
from video_annotation_utils import (
    create_clip_file_name,
    get_clip_action_label,
    extract_clip,
    extract_negative_samples_per_file,
)


def main(
    annotation_filepath,
    has_header,
    video_dir,
    clip_dir,
    label_filepath,
    clip_format,
    clip_margin,
    clip_length,
):
    # set pandas display
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    if has_header:
        skiprows = 1
    else:
        skiprows = 0

    # read in the start time and end time of the clips while removing the records with no label related
    video_info_df = pd.read_csv(annotation_filepath, skiprows=skiprows)
    video_info_df = video_info_df.loc[video_info_df["metadata"] != "{}"]

    # create clip file name and label
    video_info_df["clip_file_name"] = video_info_df.apply(
        lambda x: create_clip_file_name(x, clip_file_format=clip_format),
        axis=1,
    )
    video_info_df["clip_action_label"] = video_info_df.apply(
        lambda x: get_clip_action_label(x), axis=1
    )

    # remove the clips with action label as '_DEFAULT'
    video_info_df = video_info_df.loc[
        video_info_df["clip_action_label"] != "_DEFAULT"
    ]

    # script-input
    video_info_df.apply(lambda x: extract_clip(x, video_dir, clip_dir), axis=1)

    # write the label
    video_info_df[["clip_file_name", "clip_action_label"]].to_csv(
        label_filepath, index=False
    )

    # Extract negative samples
    # add column with file name
    video_info_df["video_file"] = video_info_df.apply(
        lambda x: ast.literal_eval(x.file_list)[0], axis=1
    )
    negative_clip_dir = os.path.join(clip_dir, "negative_samples")
    video_file_list = list(video_info_df["video_file"].unique())
    negative_sample_info_df = pd.DataFrame()
    for video_file in video_file_list:
        res_df = extract_negative_samples_per_file(
            video_file,
            video_dir,
            video_info_df,
            negative_clip_dir,
            clip_format,
            ignore_clip_length=clip_margin,
            clip_length=clip_length,
            skip_clip_length=clip_margin,
        )

        negative_sample_info_df = negative_sample_info_df.append(res_df)

    negative_sample_info_df.to_csv(
        os.path.join(negative_clip_dir, "negative_clip_info.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-A", "--annotation_filepath", help="CSV filepath from the annotator"
    )
    parser.add_argument(
        "-H",
        "--has_header",
        help="Set if the annotation file has header",
        action="store_true",
    )
    parser.add_argument("-I", "--input_dir", help="Input video dir")
    parser.add_argument(
        "-O",
        "--output_dir",
        help="Output dir where the extracted clips will be stored",
        default="./outputs",
    )
    parser.add_argument(
        "-L",
        "--label_filepath",
        help="Path where the label csv will be stored",
        default="./outputs/labels.csv",
    )
    parser.add_argument("-F", "--clip_format", default="mp4")
    parser.add_argument(
        "-M",
        "--clip_margin",
        type=float,
        help="The length around the positive samples to be ignored for negative sampling",
        default=3.0,
    )
    parser.add_argument(
        "-T",
        "--clip_length",
        type=float,
        help="The length of negative samples to extract",
        default=2.0,
    )
    args = parser.parse_args()

    main(
        annotation_filepath=args.annotation_filepath,
        has_header=args.has_header,
        video_dir=args.input_dir,
        clip_dir=args.output_dir,
        label_filepath=args.label_filepath,
        clip_format=args.clip_format,
        clip_margin=args.clip_margin,
        clip_length=args.clip_length,
    )
