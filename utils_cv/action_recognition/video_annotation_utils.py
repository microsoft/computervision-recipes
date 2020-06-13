# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
import os
import subprocess
from collections import defaultdict
import random
import numpy as np
import pandas as pd


# transform the encoded video:
def video_format_conversion(video_path, output_path, h264_format=False):
    """
    Encode video in a different format.

    :param video_path: str.
        Path to input video
    :param output_path: str.
        Path where converted video will be written to.
    :param h264_format: boolean.
        Set to true to save time if input is in h264_format.
    :return: None.
    """

    if not h264_format:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                video_path,
                "-c",
                "copy",
                "-map",
                "0",
                output_path,
            ]
        )
    else:
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vcodec", "libx264", output_path]
        )


def parse_video_file_name(row):
    """
    Extract file basename from file path

    :param row: Pandas.Series.
        One row of the video annotation output from the VIA tool.
    :return: str.
        The file basename
    """
    video_file = ast.literal_eval(row.file_list)[0]
    return os.path.basename(video_file).replace("%20", " ")


def read_classes_file(classes_filepath):
    """
    Read file that maps class names to class IDs. The file should be in the format:
        ActionName1 0
        ActionName2 1

    :param classes_filepath: str
        The filepath of the classes file
    :return: dict
        Mapping of class names to class IDs
    """
    classes = {}
    with open(classes_filepath) as class_file:
        for line in class_file:
            class_name, class_id = line.split(" ")
            classes[class_name] = class_id.rstrip()
    return classes


def create_clip_file_name(row, clip_file_format="mp4"):
    """
    Create the output clip file name.

    :param row: pandas.Series.
        One row of the video annotation output from the VIA tool.
        This function requires the output from VIA tool contains a column '# CSV_HEADER = metadata_id'.
    :param clip_file_format: str.
        The format of the output clip file.
    :return: str.
        The output clip file name.
    """
    # video_file = ast.literal_eval(row.file_list)[0]
    video_file = os.path.splitext(row["file_list"])[0]
    clip_id = row["# CSV_HEADER = metadata_id"]
    clip_file = "{}_{}.{}".format(video_file, clip_id, clip_file_format)
    return clip_file


def get_clip_action_label(row):
    """
    Get the action label of the positive clips.
    This function requires the output from VIA tool contains a column 'metadata'.

    :param row: pandas.Series.
        One row of the video annotation output.
    :return: str.
    """
    label_dict = ast.literal_eval(row.metadata)
    track_key = list(label_dict.keys())[0]
    return label_dict[track_key]


def _extract_clip_ffmpeg(
    start_time, duration, video_path, clip_path, ffmpeg_path=None
):
    """
    Using ffmpeg to extract clip from the video based on the start time and duration of the clip.

    :param start_time: float.
        The start time of the clip.
    :param duration: float.
        The duration of the clip.
    :param video_path: str.
        The path of the input video.
    :param clip_path: str.
        The path of the output clip.
    :param ffmpeg_path: str.
        The path of the ffmpeg. This is optional, which you could use when you have not added the
        ffmpeg to the path environment variable.
    :return: None.
    """

    subprocess.run(
        [
            os.path.join(ffmpeg_path, "ffmpeg")
            if ffmpeg_path is not None
            else "ffmpeg",
            "-ss",
            str(start_time),
            "-i",
            video_path,
            "-t",
            str(duration),
            clip_path,
            "-codec",
            "copy",
            "-y",
        ]
    )


def extract_clip(row, video_dir, clip_dir, ffmpeg_path=None):
    """
    Extract the postivie clip based on a row of the output annotation file.

    :param row: pandas.Series.
        One row of the video annotation output.
    :param video_dir: str.
        The directory of the input videos.
    :param clip_dir: str.
        The directory of the output positive clips.
    :param ffmpeg_path: str.
        The path of the ffmpeg. This is optional, which you could use when you have not added the
        ffmpeg to the path environment variable.
    :return: None.
    """

    if not os.path.exists(clip_dir):
        os.makedirs(clip_dir)

    # there are two different output of the csv from the VIA showing the annotation start and end
    # (1) in two columns: temporal_segment_start and temporal_segment_end
    # (2) in one column: temporal_coordinates
    if "temporal_segment_start" in row.index:
        start_time = row.temporal_segment_start
        if "temporal_segment_end" not in row.index:
            raise ValueError(
                "There is no column named 'temporal_segment_end'. Cannot get the full details "
                "of the action temporal intervals."
            )
        end_time = row.temporal_segment_end
    elif "temporal_coordinates" in row.index:
        start_time, end_time = ast.literal_eval(row.temporal_coordinates)
    else:
        raise Exception("There is no temporal information in the csv.")

    clip_sub_dir = os.path.join(clip_dir, row.clip_action_label)
    if not os.path.exists(clip_sub_dir):
        os.makedirs(clip_sub_dir)

    duration = end_time - start_time
    video_file = row.file_list
    video_path = os.path.join(video_dir, video_file)
    clip_file = row.clip_file_name
    clip_path = os.path.join(clip_sub_dir, clip_file)

    # skip if video already extracted
    if os.path.exists(clip_path):
        print("Extracted clip already exists. Skipping extraction.")
        return

    if not os.path.exists(video_path):
        raise ValueError(
            "The video path '{}' is not valid.".format(video_path)
        )

    # ffmpeg -ss 9.222 -i youtube.mp4 -t 0.688 tmp.mp4 -codec copy -y
    _extract_clip_ffmpeg(
        start_time, duration, video_path, clip_path, ffmpeg_path
    )


def get_video_length(video_file_path):
    """
    Get the video length in milliseconds.

    :param video_file_path: str.
        The path of the video file.
    :return: (str, str).
        Tuple of video duration (in string), and error message of the ffprobe command if any.
    """
    cmd_list = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_file_path,
    ]
    result = subprocess.run(
        cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if len(result.stderr) > 0:
        raise RuntimeError(result.stderr)

    return float(result.stdout)


def check_interval_overlaps(clip_start, clip_end, interval_list):
    """
    Check whether a clip overlaps any intervals from a list of intervals.

    param clip_start: float
        Time in seconds of the start of the clip.
    param clip_end: float
        Time in seconds of the end of the clip.
    param interval_list: list of tuples (float, float)
        List of time intervals
    return: Boolean
        True if the clip overlaps any of the intervals in interval list.
    """
    overlapping = False
    for interval in interval_list:
        if (clip_start < interval[1]) and (clip_end > interval[0]):
            overlapping = True
    return overlapping


def _merge_temporal_interval(temporal_interval_list):
    """
    Merge the temporal intervals in the input temporal interval list. This is for situations
    when different actions have overlap temporal interval. e.g if the input temporal interval list
    is [(1.0, 3.0), (2.0, 4.0)], then [(1.0, 4.0)] will be returned.

    :param temporal_interval_list: list of tuples.
        List of tuples with (temporal interval start time, temporal interval end time).
    :return: list of tuples.
        The merged temporal interval list.
    """
    # sort by the temporal interval start
    temporal_interval_list_sorted = sorted(
        temporal_interval_list, key=lambda x: x[0]
    )
    i = 0

    while i < len(temporal_interval_list_sorted) - 1:
        a1, b1 = temporal_interval_list_sorted[i]
        a2, b2 = temporal_interval_list_sorted[i + 1]
        if a2 <= b1:
            del temporal_interval_list_sorted[i]
            temporal_interval_list_sorted[i] = [a1, max(b1, b2)]
        else:
            i += 1
    return temporal_interval_list_sorted


def _split_interval(
    interval,
    left_ignore_clip_length,
    right_ignore_clip_length,
    clip_length,
    skip_clip_length=0,
):
    """
    Split the negative sample interval into the sub-intervals which will serve as the start and end of
    the negative sample clips.

    :param interval: tuple of (float, float).
        Tuple of start and end of the negative sample interval.
    :param left_ignore_clip_length: float.
        The clip length to ignore in the left/start of the interval. This is used to avoid creating
        negative sample clips with edges too close to positive samples. The same applies to right_ignore_clip_length.
    :param right_ignore_clip_length: float.
        The clip length to ignore in the right/end of the interval.
    :param clip_length: float.
        The clip length of the created negative clips.
    :param skip_clip_length: float.
        The skipped video length between two negative samples, this can be used to reduce the
        number of the negative samples.
    :return: list of tuples.
        List of start and end time of the negative clips.
    """
    left, right = interval
    if (left_ignore_clip_length + right_ignore_clip_length) >= (right - left):
        return []
    new_left = left + left_ignore_clip_length
    new_right = right - right_ignore_clip_length

    if new_right - new_left < clip_length:
        return []

    interval_start_list = np.arange(
        new_left, new_right, clip_length + skip_clip_length
    )
    interval_end_list = interval_start_list + clip_length

    if interval_end_list[-1] > new_right:
        interval_start_list = interval_start_list[:-1]
        interval_end_list = interval_end_list[:-1]

    res = list(zip(list(interval_start_list), list(interval_end_list)))
    return res


def _split_interval_list(
    interval_list,
    left_ignore_clip_length,
    right_ignore_clip_length,
    clip_length,
    skip_clip_length=0,
):
    """
    Taking the interval list of the eligible negative sample time intervals, return the list of the
    start time and the end time of the negative clips.

    :param interval_list: list of tuples.
        List of the tuples containing the start time and end time of the eligible negative
        sample time intervals.
    :param left_ignore_clip_length: float.
        See split_interval.
    :param right_ignore_clip_length: float.
        See split_interval.
    :param clip_length: float.
        See split_interval.
    :param skip_clip_length: float.
        See split_interval
    :return: list of tuples.
        List of start and end time of the negative clips.
    """
    interval_res = []
    for i in range(len(interval_list)):
        interval_res += _split_interval(
            interval_list[i],
            left_ignore_clip_length=left_ignore_clip_length,
            right_ignore_clip_length=right_ignore_clip_length,
            clip_length=clip_length,
            skip_clip_length=skip_clip_length,
        )
    return interval_res


def extract_contiguous_negative_clips(
    video_file,
    video_dir,
    video_info_df,
    negative_clip_dir,
    clip_format,
    no_action_class,
    ignore_clip_length,
    clip_length,
    skip_clip_length=0,
    ffmpeg_path=None,
):
    """
    Extract the negative sample for a single video file.

    :param video_file: str.
        The name of the input video file.
    :param video_dir: str.
        The directory of the input video.
    :param video_info_df: pandas.DataFrame.
        The data frame which contains the video annotation output.
    :param negative_clip_dir: str.
        The directory of the output negative clips.
    :param clip_format: str.
        The format of the output negative clips.
    :param ignore_clip_length: float.
        The clip length to ignore in the left/start of the interval. This is used to avoid creating
        negative sample clips with edges too close to positive samples.
    :param clip_length: float.
        The clip length of the created negative clips.
    :param ffmpeg_path: str.
        The path of the ffmpeg. This is optional, which you could use when you have not added the
        ffmpeg to the path environment variable.
    :param skip_clip_length: float.
        The skipped video length between two negative samples, this can be used to reduce the
        number of the negative samples.
    :return: pandas.DataFrame.
        The data frame which contains start and end time of the negative clips.
    """

    # get the length of the video
    video_file_path = os.path.join(video_dir, video_file)
    video_duration = get_video_length(video_file_path)

    # get the actions intervals
    if "temporal_coordinates" in video_info_df.columns:
        temporal_interval_series = video_info_df.loc[
            video_info_df["file_list"] == video_file, "temporal_coordinates"
        ]
        temporal_interval_list = temporal_interval_series.apply(
            lambda x: ast.literal_eval(x)
        ).tolist()
    elif "temporal_segment_start" in video_info_df.columns:
        video_start_list = video_info_df.loc[
            video_info_df["file_list"] == video_file, "temporal_segment_start"
        ].tolist()
        video_end_list = video_info_df.loc[
            video_info_df["file_list"] == video_file, "temporal_segment_end"
        ].tolist()
        temporal_interval_list = list(zip(video_start_list, video_end_list))
    else:
        raise Exception("There is no temporal information in the csv.")

    if not all(
        len(temporal_interval) % 2 == 0
        for temporal_interval in temporal_interval_list
    ):
        raise ValueError(
            "There is at least one time interval "
            "in {} having only one end point.".format(
                str(temporal_interval_list)
            )
        )

    temporal_interval_list = _merge_temporal_interval(temporal_interval_list)
    negative_sample_interval_list = (
        [0.0]
        + [t for interval in temporal_interval_list for t in interval]
        + [video_duration]
    )

    negative_sample_interval_list = [
        [
            negative_sample_interval_list[2 * i],
            negative_sample_interval_list[2 * i + 1],
        ]
        for i in range(len(negative_sample_interval_list) // 2)
    ]

    clip_interval_list = _split_interval_list(
        negative_sample_interval_list,
        left_ignore_clip_length=ignore_clip_length,
        right_ignore_clip_length=ignore_clip_length,
        clip_length=clip_length,
        skip_clip_length=skip_clip_length,
    )

    if not os.path.exists(negative_clip_dir):
        os.makedirs(negative_clip_dir)

    negative_clip_file_list = []
    for i, clip_interval in enumerate(clip_interval_list):
        start_time = clip_interval[0]
        duration = clip_interval[1] - clip_interval[0]
        # negative_clip_file = "{}_{}.{}".format(video_file, i, clip_file_format)

        # video_path = os.path.join(video_dir, negative_sample_file)
        video_fname = os.path.splitext(os.path.basename(video_file_path))[0]
        clip_fname = video_fname + no_action_class + str(i)
        clip_subdir_fname = os.path.join(no_action_class, clip_fname)
        negative_clip_file_list.append(clip_subdir_fname)
        _extract_clip_ffmpeg(
            start_time,
            duration,
            video_file_path,
            os.path.join(negative_clip_dir, clip_fname + "." + clip_format),
            ffmpeg_path,
        )

    return pd.DataFrame(
        {
            "negative_clip_file_name": negative_clip_file_list,
            "clip_duration": clip_interval_list,
            "video_file": video_file,
        }
    )


def extract_sampled_negative_clips(
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
):
    """
    Extract randomly sampled negative clips from a set of videos.

    param video_info_df: Pandas.DataFrame
        DataFrame containing annotated video information
    param num_negative_samples: int
        Number of negative samples to extract
    param video_files: listof str
        List of original video files
    param video_dir: str
        Directory of original videos
    param clip_dir: str
        Directory of extracted clips
    param classes: dict
        Classes dictionary
    param no_action_class: str
        Name of no action class
    param negative_clip_length: float
        Length of clips in seconds
    param clip_format: str
        Format for video files
    param label_filepath: str
        Path to the label file
    return: None
    """
    # find video lengths
    video_len = {}
    for video in video_files:
        video_len[video] = get_video_length(os.path.join(video_dir, video))
    positive_intervals = defaultdict(list)
    # get temporal interval of positive samples
    for index, row in video_info_df.iterrows():
        clip_file = row.file_list
        int_start = row.temporal_segment_start
        int_end = row.temporal_segment_end
        segment_int = (int_start, int_end)
        positive_intervals[clip_file].append(segment_int)
    clips_sampled = 0
    while clips_sampled < num_negative_samples:
        # pick random file in list of videos
        negative_sample_file = video_files[
            random.randint(0, len(video_files) - 1)
        ]
        # get video duration
        duration = video_len[negative_sample_file]
        # pick random start time for clip
        clip_start = random.uniform(0.0, duration)
        clip_end = clip_start + negative_clip_length
        if clip_end > duration:
            continue
        # check to ensure negative clip doesn't overlap a positive clip or pick another file
        if negative_sample_file in positive_intervals.keys():
            clip_positive_intervals = positive_intervals[negative_sample_file]
            if check_interval_overlaps(
                clip_start, clip_end, clip_positive_intervals
            ):
                continue
        video_path = os.path.join(video_dir, negative_sample_file)
        video_fname = os.path.splitext(negative_sample_file)[0]
        clip_fname = video_fname + no_action_class + str(clips_sampled)
        clip_subdir_fname = os.path.join(no_action_class, clip_fname)
        _extract_clip_ffmpeg(
            clip_start,
            negative_clip_length,
            video_path,
            os.path.join(clip_dir, clip_subdir_fname + "." + clip_format),
        )
        with open(label_filepath, "a") as f:
            f.write(
                '"'
                + clip_subdir_fname
                + '"'
                + " "
                + str(classes[no_action_class])
                + "\n"
            )
        clips_sampled += 1
