# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
import math
import os
import sys

sys.path.append("../")

import pandas as pd
import pytest

from video_annotation.video_annotation_utils import (
    # Usually don't test private functions.
    # But as clip-extraction results are tricky to test, we test some of the private functions here.
    _merge_temporal_interval,
    _split_interval_list,
    create_clip_file_name,
    extract_clip,
    extract_negative_samples_per_file,
    get_clip_action_label,
    get_video_length,
)


VIDEO_DIR = os.path.join("tests", "data")
SAMPLE_VIDEO1_FILE = "3173 1-7 Cold 2019-08-19_13_56_14_787.mp4"
SAMPLE_VIDEO1_PATH = os.path.join(VIDEO_DIR, SAMPLE_VIDEO1_FILE)
SAMPLE_ANNOTATION_FILE = "Unnamed-VIA Project19Sep2019_18h42m15s_export.csv"
SAMPLE_ANNOTATION_PATH = os.path.join(VIDEO_DIR, SAMPLE_ANNOTATION_FILE)
FRAME_PER_SECOND = 30


@pytest.fixture
def annotation_df():
    video_info_df = pd.read_csv(SAMPLE_ANNOTATION_PATH, skiprows=1)
    return video_info_df.loc[video_info_df["metadata"] != "{}"]


def test_create_clip_file_name(annotation_df):
    row1 = annotation_df.iloc[0]
    file1 = create_clip_file_name(row1, clip_file_format="mp4")
    assert file1 == "3173 1-7 Cold 2019-08-19_13_56_14_787.mp4_1_zCXg2CQ5.mp4"
    file2 = create_clip_file_name(row1, clip_file_format="avi")
    assert file2 == "3173 1-7 Cold 2019-08-19_13_56_14_787.mp4_1_zCXg2CQ5.avi"


def test_get_clip_action_label(annotation_df):
    row1 = annotation_df.iloc[0]
    assert get_clip_action_label(row1) == "1.action_1"


def test_extract_clip(annotation_df, tmp_path):
    row1 = annotation_df.iloc[0].copy()
    row1["clip_action_label"] = get_clip_action_label(row1)
    row1["clip_file_name"] = create_clip_file_name(
        row1, clip_file_format="mp4"
    )
    extract_clip(
        row=row1, video_dir=VIDEO_DIR, clip_dir=tmp_path, ffmpeg_path=None
    )
    output_clip_path = os.path.join(
        tmp_path, row1["clip_action_label"], row1["clip_file_name"]
    )

    # Test if extracted positive clip length is the same as the annotated segment length
    assert (
        abs(
            get_video_length(output_clip_path)
            - (row1.temporal_segment_end - row1.temporal_segment_start)
        )
        <= 1 / FRAME_PER_SECOND
    )


def test_extract_negative_samples_per_file(annotation_df, tmp_path):
    """TODO This function should test two things which are missing now:
    1. assert if the extracted negative samples are not overlapped with any positive samples
    2. assert the number of extracted negative samples are correct
    """
    video_df = annotation_df.copy()
    video_df["video_file"] = video_df.apply(
        lambda x: ast.literal_eval(x.file_list)[0], axis=1
    )
    clip_length = 2
    extract_negative_samples_per_file(
        video_file=SAMPLE_VIDEO1_FILE,
        video_dir=VIDEO_DIR,
        video_info_df=video_df,
        negative_clip_dir=tmp_path,
        clip_file_format="mp4",
        ignore_clip_length=0,
        clip_length=clip_length,
        ffmpeg_path=None,
        skip_clip_length=0,
    )
    for i in range(4):
        negative_clip_length = get_video_length(
            os.path.join(tmp_path, "{}_{}.mp4".format(SAMPLE_VIDEO1_FILE, i))
        )
        assert abs(negative_clip_length - clip_length) <= 1 / FRAME_PER_SECOND


def test_get_video_length():
    assert get_video_length(SAMPLE_VIDEO1_PATH) == 18.719


def test_merge_temporal_interval():
    interval_list1 = [(1, 2.5), (1.5, 2), (0.5, 1.5)]
    merged_interval_list1 = _merge_temporal_interval(interval_list1)
    assert merged_interval_list1 == [[0.5, 2.5]]

    interval_list2 = [(-1.1, 0), (0, 1.2), (4.5, 7), (6.8, 8.5)]
    merged_interval_list2 = _merge_temporal_interval(interval_list2)
    assert merged_interval_list2 == [[-1.1, 1.2], [4.5, 8.5]]


def _float_tuple_close(input_tuple1, input_tuple2):
    return all(
        math.isclose(input_tuple1[i], input_tuple2[i])
        for i in range(len(input_tuple1))
    )


def _float_tuple_list_close(input_tuple_list1, input_tuple_list2):
    return all(
        _float_tuple_close(input_tuple_list1[i], input_tuple_list2[i])
        for i in range(len(input_tuple_list1))
    )


def test_split_interval_list():
    interval_list1 = [(0.5, 3.0), (5.0, 9.0)]
    res1 = _split_interval_list(
        interval_list1,
        left_ignore_clip_length=0.3,
        right_ignore_clip_length=0.5,
        clip_length=0.6,
        skip_clip_length=0.1,
    )
    assert _float_tuple_list_close(
        res1,
        [
            (0.8, 1.4),
            (1.5, 2.1),
            (5.3, 5.9),
            (6.0, 6.6),
            (6.7, 7.3),
            (7.4, 8.0),
        ],
    )
