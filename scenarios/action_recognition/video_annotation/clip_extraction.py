# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# prerequisite:
# (1) download and extract ffmpeg: https://github.com/adaptlearning/adapt_authoring/wiki/Installing-FFmpeg
# (2) make sure the ffmpeg is in your system's env variable: path

"""
Extracts clips from videos stored in video_dir. Clips are defined from the annotated actions recorded in
annotation_filepath, which is the raw output csv from the VIA video annotator tool. This csv should have the format:

    # Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via)				
    # CSV_HEADER = metadata_id,	file_list,	temporal_segment_start,	temporal_segment_end,	metadata

The script will generate clips for actions (classes) that appear in label_filepath, a mapping of class labels
to class ID numbers. The label_filepath file should have the format:

    Action1 0
    Action2 1
    Action3 2

Optionally, "negative" examples can be extracted in which no action-of-interest occurs. To generate negative
examples, the name to give to the negative class must be provided with no_action_class. Negative clips can be
extracted in two ways: either all contiguous non-overlapping negative clips can be extracted or a specified
number of negative examples can be randomly sampled. This behaviour can be controlled using the `contiguous`
flag. The sample_annotated_only flag can be used to specify whether negative samples are extracted from any
video in video_dir, or only those with annotations. The script outputs clips into subdirectories of clip_dir 
specific to each class and generates a label file that maps each filename to the clip's class label.
"""

import argparse
import ast
import os
import sys
import pandas as pd

sys.path.append("../../../utils_cv/action_recognition/")
from video_annotation_utils import (
    parse_video_file_name,
    read_classes_file,
    create_clip_file_name,
    get_clip_action_label,
    extract_clip,
    extract_contiguous_negative_clips,
    extract_sampled_negative_clips
)


def main(
    annotation_filepath,
    video_dir,
    clip_dir,
    classes_filepath,
    label_filepath,
    clip_format,
    no_action_class,
    contiguous,
    negative_clip_length,
    negative_clip_margin,
    sample_annotated_only,
    num_negative_samples,
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

    # create clip file name and label
    video_info_df["clip_file_name"] = video_info_df.apply(
        lambda x: create_clip_file_name(x, clip_file_format=clip_format),
        axis=1,
    )

    video_info_df["clip_action_label"] = video_info_df.apply(
        lambda x: get_clip_action_label(x), axis=1
    )

    # read list of classes
    classes = read_classes_file(classes_filepath)
    if no_action_class is not None:
        if no_action_class not in classes:
            raise Exception("no_action_class does not appear in list of classes.")

    # filter annotations to only include actions that appear in the classes file
    video_info_df = video_info_df[video_info_df["clip_action_label"].isin(classes.keys())]

    # extract all positive examples
    video_info_df.apply(lambda x: extract_clip(x, video_dir, clip_dir), axis=1)

    # write the labels for positive examples
    video_info_df["clip_file_path"] = video_info_df.apply(lambda row: os.path.join(row.clip_action_label, row.clip_file_name), axis=1)
    video_info_df["clip_file_path"] = video_info_df["clip_file_path"].apply(lambda x: os.path.splitext(x)[0])
    video_info_df["clip_class_id"] = video_info_df["clip_action_label"].apply(lambda x: classes[x])
    video_info_df[["clip_file_path", "clip_class_id"]].to_csv(
        label_filepath, header=None, index=False, sep=' '
    )

    # Extract negative samples if required
    if no_action_class:
        negative_clip_dir = os.path.join(clip_dir, no_action_class)
        if not os.path.exists(negative_clip_dir):
            os.makedirs(negative_clip_dir)
        if contiguous:
            video_files = list(video_info_df["file_list"].unique())
            negative_sample_info_df = pd.DataFrame()
            for video_file in video_files:
                res_df = extract_contiguous_negative_clips(
                    video_file,
                    video_dir,
                    video_info_df,
                    negative_clip_dir,
                    clip_format,
                    no_action_class,
                    ignore_clip_length=negative_clip_margin,
                    clip_length=negative_clip_length,
                    skip_clip_length=negative_clip_margin,
                )
                negative_sample_info_df = negative_sample_info_df.append(res_df)
            with open(label_filepath, 'a') as f:
                for index, row in negative_sample_info_df.iterrows():
                    f.write("\""+row.negative_clip_file_name+"\""+" "+str(classes[no_action_class])+"\n")
        
        else:
            # get list of original video files
            video_files = os.listdir(video_dir)

            if sample_annotated_only:
                video_files = list(set(video_info_df["file_list"]) & set(video_files))
            
            extract_sampled_negative_clips(
                video_info_df,
                num_negative_samples,
                video_files,
                video_dir,
                clip_dir,
                classes,
                no_action_class,
                negative_clip_length,
                clip_format,
                label_filepath,
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_filepath", help="CSV filepath from the annotator", required=True,
    )
    parser.add_argument(
        "--video_dir", help="Input video dirictory", required=True
    )
    parser.add_argument(
        "--clip_dir",
        help="Output directory where the extracted clips will be stored",
        required=True,
    )
    parser.add_argument(
        "--classes_filepath", help="Path to file defining classes and class IDs", required=True
    )
    parser.add_argument(
        "--label_filepath",
        help="Path where the label file will be stored",
        required=True,
    )
    parser.add_argument(
        "--clip_format", default="mp4"
    )
    parser.add_argument(
        "--no_action_class",
        help="Label for the no action class. Provide this argument to create negative examples."
    )
    parser.add_argument(
        "--contiguous",
        help="Set to true to extract all non-overlapping negative samples. Otherwise extract num_negative_samples randomly sampled negative clips.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--negative_clip_length",
        type=float,
        help="The length of negative samples to extract",
        default=2.0,
    )
    parser.add_argument(
        "--negative_clip_margin",
        type=float,
        help="The length around the positive samples to be ignored for negative sampling",
        default=3.0,
    )
    parser.add_argument(
        "--sample_annotated_only",
        help="Source negative clips only from videos that have at least one positive action (only for non-contiguous samples)",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--num_negative_samples",
        type=float,
        help="The number of negative clips to sample. This only applies to non-contiguous sampling.",
        default=0.0
    )
    args = parser.parse_args()

    main(
        annotation_filepath=args.annotation_filepath,
        video_dir=args.video_dir,
        clip_dir=args.clip_dir,
        classes_filepath=args.classes_filepath,
        label_filepath=args.label_filepath,
        clip_format=args.clip_format,
        no_action_class=args.no_action_class,
        contiguous=args.contiguous,
        negative_clip_length=args.negative_clip_length,
        negative_clip_margin=args.negative_clip_margin,
        sample_annotated_only=args.sample_annotated_only,
        num_negative_samples=args.num_negative_samples
    )
