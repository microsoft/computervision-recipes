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
import random
from collections import defaultdict

import pandas as pd

sys.path.append("lib")
from video_annotation_utils import (
    parse_video_file_name,
    read_classes_file,
    create_clip_file_name,
    get_clip_action_label,
    extract_clip,
    extract_negative_samples_per_file,
    get_video_length,
    check_interval_overlaps,
    _extract_clip_ffmpeg,
)


def main(
    annotation_filepath,
    classes_filepath,
    video_dir,
    clip_dir,
    label_filepath,
    clip_format,
    clip_margin,
    clip_length,
    contiguous,
    num_negative_samples,
    filter_positive_videos
):
    # set pandas display
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 1000)

    # read in the start time and end time of the clips while removing the records with no label related
    video_info_df = pd.read_csv(annotation_filepath, skiprows=1)
    video_info_df = video_info_df.loc[video_info_df["metadata"] != "{}"]


    video_info_df["file_list"] = video_info_df.apply(
        lambda x: parse_video_file_name(x),
        axis=1,
    )

    # get list of original video files
    # TODO: finish annotation for all files
    video_files = os.listdir(video_dir)[:320]

    if filter_positive_videos:
        video_files = list(set(video_info_df["file_list"]) & set(video_files))

    # find video lengths
    video_len = {}
    for video in video_files:
        video_len[video] = get_video_length(os.path.join(video_dir, video))
    
    # create clip file name and label
    video_info_df["clip_file_name"] = video_info_df.apply(
        lambda x: create_clip_file_name(x, clip_file_format=clip_format),
        axis=1,
    )

    video_info_df["clip_action_label"] = video_info_df.apply(
        lambda x: get_clip_action_label(x), axis=1
    )

    # read list of classes
    if classes_filepath is not None:
        classes = read_classes_file(classes_filepath)

    # filter classes
    video_info_df = video_info_df[video_info_df["clip_action_label"].isin(classes.keys())]

    # extract all positive examples
    video_info_df.apply(lambda x: extract_clip(x, video_dir, clip_dir), axis=1)

    # write the labels
    video_info_df["clip_file_path"] = video_info_df.apply(lambda row: os.path.join(row.clip_action_label, row.clip_file_name), axis=1)
    video_info_df["clip_file_path"] = video_info_df["clip_file_path"].apply(lambda x: os.path.splitext(x)[0])
    video_info_df["clip_class_id"] = video_info_df["clip_action_label"].apply(lambda x: classes[x])
    video_info_df[["clip_file_path", "clip_class_id"]].to_csv(
        label_filepath, header=None, index=False, sep=' '
    )


    # Extract negative samples
    if contiguous:
        negative_clip_dir = os.path.join(clip_dir, "NoAction")
        video_file_list = list(video_info_df["file_list"].unique())
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
        with open(label_filepath, 'a') as f:
            for index, row in negative_sample_info_df.iterrows():
                f.write("\""+row.negative_clip_file_list+"\""+" "+str(classes["NoAction"])+"\n")
    
    else:
        # get temporal interval of positive samples
        positive_intervals = defaultdict(list)
        for index, row in video_info_df.iterrows():
            clip_file = row.file_list
            int_start = row.temporal_segment_start
            int_end = row.temporal_segment_end
            segment_int = (int_start, int_end)
            positive_intervals[clip_file].append(segment_int)
        negative_sample_dir = os.path.join(clip_dir, "NoAction")
        if not os.path.exists(negative_sample_dir):
            os.makedirs(negative_sample_dir)
        clips_sampled = 0
        while clips_sampled < num_negative_samples:
            # pick random file in list of videos
            negative_sample_file = video_files[random.randint(0, len(video_files)-1)]
            # get video duration
            duration = video_len[negative_sample_file]
            # pick random start time for clip
            clip_start = random.uniform(0.0, duration)
            clip_end = clip_start + clip_length
            if clip_end > duration:
                continue
            # check to ensure negative clip doesn't overlap a positive clip or pick another file
            if negative_sample_file in positive_intervals.keys():
                clip_positive_intervals = positive_intervals[negative_sample_file]
                if check_interval_overlaps(clip_start, clip_end, clip_positive_intervals):
                    continue
            video_path = os.path.join(video_dir, negative_sample_file)
            video_fname = os.path.splitext(negative_sample_file)[0]
            clip_fname = video_fname+"NoAction"+str(clips_sampled)
            clip_subdir_fname = os.path.join("NoAction", clip_fname)
            _extract_clip_ffmpeg(
                clip_start, clip_length, video_path, os.path.join(clip_dir, clip_subdir_fname+"."+clip_format),
            )
            with open(label_filepath, 'a') as f:
                f.write("\""+clip_subdir_fname+"\""+" "+str(classes["NoAction"])+"\n")
            clips_sampled += 1


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
    parser.add_argument(
        "-C",
        "--classes",
        help="List of classes to generate",
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
    parser.add_argument(
        "-c",
        "--contiguous",
        help="Set to true to extract all non-overlapping negative samples. Otherwise extract num_negative_samples",
        action="store_true",
    )
    parser.add_argument(
        "-m",
        "--num_negative_samples",
        type=float,
        help="The number of random negative samples to extract",
        default=0.0
    )
    args = parser.parse_args()

    main(
        annotation_filepath="b2510Dec2019_11h52m31s_export.csv",
        classes_filepath="classes.txt",
        has_header=True,
        video_dir="videos_original",
        clip_dir="videos_processed",
        label_filepath="labels.txt",
        clip_format="mp4",
        clip_margin=3.0,
        clip_length=2.0,
        contiguous=False,
        num_negative_samples=500.0,
        filter_positive_videos=True
    )

    # main(
    #     annotation_filepath=args.annotation_filepath,
    #     has_header=args.has_header,
    #     video_dir=args.input_dir,
    #     clip_dir=args.output_dir,
    #     label_filepath=args.label_filepath,
    #     clip_format=args.clip_format,
    #     clip_margin=args.clip_margin,
    #     clip_length=args.clip_length,
    # )
