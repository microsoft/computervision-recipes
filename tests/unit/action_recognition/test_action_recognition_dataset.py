# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import pytest
import torchvision
import torch

from utils_cv.common.misc import Config
from utils_cv.action_recognition.dataset import (
    VideoRecord,
    get_transforms,
    DEFAULT_MEAN,
    DEFAULT_STD,
    get_default_tfms_config,
    VideoDataset,
)


def check_VideoRecord(record: VideoRecord, ar_vid_path: str) -> None:
    """ Checks that property methods work. """
    assert record.path is ar_vid_path
    assert record.label == 0
    assert record.label_name in ("cooking", None)


def test_VideoRecord(ar_vid_path) -> None:
    """ Test the video record initialization. """
    correct_input_one = [ar_vid_path, 0, "cooking"]
    check_VideoRecord(VideoRecord(correct_input_one), ar_vid_path)

    correct_input_two = [ar_vid_path, 0]
    check_VideoRecord(VideoRecord(correct_input_two), ar_vid_path)


def test_VideoRecord_invalid(ar_vid_path) -> None:
    """ Test the video record initialization failure. """
    incorrect_inputs = [
        [ar_vid_path, "0", "cooking", "extra"],
        [ar_vid_path],
        [ar_vid_path, "cooking", 0],
        ["ar_vid_path, 0, cooking"],
        "ar_vid_path, 0, cooking",
    ]
    for inp in incorrect_inputs:
        with pytest.raises(Exception):
            VideoRecord(inp)


def test_get_transforms() -> None:
    """ Test the transforms function. """
    train_tfms = get_transforms(train=True)
    assert isinstance(train_tfms, torchvision.transforms.Compose)

    test_tfms = get_transforms(train=False)
    assert isinstance(test_tfms, torchvision.transforms.Compose)

    conf = Config(
        dict(
            input_size=300,
            im_scale=128,
            resize_keep_ratio=True,
            random_crop=True,
            random_crop_scales=True,
            flip_ratio=0.5,
            mean=DEFAULT_MEAN,
            std=DEFAULT_STD,
        )
    )
    custom_tfms = get_transforms(tfms_config=conf)
    assert isinstance(custom_tfms, torchvision.transforms.Compose)


def test_get_default_tfms_config() -> None:
    """ Test the function that provides basic defaults for train/test. """
    train_default_tfms = get_default_tfms_config(train=True)
    assert train_default_tfms.flip_ratio == 0.5
    assert train_default_tfms.random_crop is True
    assert train_default_tfms.random_crop_scales == (0.6, 1.0)
    assert isinstance(train_default_tfms, Config)

    test_default_tfms = get_default_tfms_config(train=False)
    assert test_default_tfms.flip_ratio == 0.0
    assert test_default_tfms.random_crop is False
    assert test_default_tfms.random_crop_scales is None
    assert isinstance(test_default_tfms, Config)


def test_VideoDataset(ar_milk_bottle_path) -> None:
    """ Test the initialization of the video dataset. """
    dataset = VideoDataset(ar_milk_bottle_path)
    assert isinstance(dataset.train_dl, torch.utils.data.DataLoader)
    assert isinstance(dataset.test_dl, torch.utils.data.DataLoader)
    assert len(dataset) == 60
    assert len(dataset.train_ds) == 45
    assert len(dataset.test_ds) == 15

    # test if train_pct is altered
    dataset = VideoDataset(ar_milk_bottle_path, train_pct=0.5)
    assert len(dataset) == 60
    assert len(dataset.train_ds) == 30
    assert len(dataset.test_ds) == 30


def test_VideoDataset_split_file(
    ar_milk_bottle_path, ar_milk_bottle_split_files,
) -> None:
    """ Tests VideoDataset initializing using split file. """
    dataset = VideoDataset(
        ar_milk_bottle_path,
        train_split_file=ar_milk_bottle_split_files[0],
        test_split_file=ar_milk_bottle_split_files[1],
    )

    assert len(dataset) == 60
    assert len(dataset.train_ds) == 40
    assert len(dataset.test_ds) == 20


def test_VideoDataset_show_batch(ar_milk_bottle_dataset) -> None:
    """ Tests the show batch functionality. """
    # test base case
    ar_milk_bottle_dataset.show_batch()

    # test with set rows
    ar_milk_bottle_dataset.show_batch(rows=3)

    # test with train_or_test == "test"
    ar_milk_bottle_dataset.show_batch(train_or_test="test")
