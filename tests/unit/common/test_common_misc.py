# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from pathlib import Path
from PIL import ImageFont

from fastai.vision import ImageList
from utils_cv.common.gpu import db_num_workers
from utils_cv.common.misc import copy_files, set_random_seed, get_font, Config


def test_set_random_seed(tiny_ic_data_path):
    # check two data batches are the same after seeding
    set_random_seed(1)
    first_data = (
        ImageList.from_folder(tiny_ic_data_path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform()
        .databunch(bs=5, num_workers = db_num_workers())
        .normalize()
    )
    first_batch = first_data.one_batch()

    set_random_seed(1)
    second_data = (
        ImageList.from_folder(tiny_ic_data_path)
        .split_by_rand_pct()
        .label_from_folder()
        .transform()
        .databunch(bs=5, num_workers = db_num_workers())
        .normalize()
    )
    second_batch = second_data.one_batch()
    assert first_batch[1].tolist() == second_batch[1].tolist()


def test_copy_files(tmp):
    parent = os.path.join(tmp, "parent")
    child = os.path.join(parent, "child")
    dst = os.path.join(tmp, "dst")
    os.makedirs(parent)
    os.makedirs(child)
    os.makedirs(dst)

    file_in_child = Path(os.path.join(child, "file_in_child.txt"))
    file_in_child.touch()

    copy_files(file_in_child, dst)
    assert os.path.isfile(os.path.join(dst, "file_in_child.txt"))

    file_in_parent = Path(os.path.join(parent, "file_in_parent.txt"))
    file_in_parent.touch()

    copy_files([file_in_child, file_in_parent], dst)
    assert os.path.isfile(os.path.join(dst, "file_in_parent.txt"))

    # Check if the subdir is inferred
    copy_files([file_in_child, file_in_parent], dst, infer_subdir=True)
    dst_child = os.path.join(dst, "child")
    assert os.path.isdir(dst_child)
    assert os.path.isfile(os.path.join(dst_child, "file_in_child.txt"))
    assert not os.path.isfile(os.path.join(dst_child, "file_in_parent.txt"))

    # Check if the original files are removed
    copy_files([file_in_child, file_in_parent], dst, remove=True)
    assert not os.path.isfile(file_in_parent)
    assert not os.path.isfile(file_in_child)


def test_get_font():
    font = get_font(size=12)
    assert (
        type(font) == ImageFont.FreeTypeFont
        or type(font) == ImageFont.ImageFont
    )


def test_Config():
    # test dictionary wrapper to make sure keys can be accessed as attributes
    cfg = Config({"lr": 0.01, "momentum": 0.95})
    assert cfg.lr == 0.01 and cfg.momentum == 0.95
    cfg = Config(lr=0.01, momentum=0.95)
    assert cfg.lr == 0.01 and cfg.momentum == 0.95
    cfg = Config({"lr": 0.01}, momentum=0.95)
    assert cfg.lr == 0.01 and cfg.momentum == 0.95
    cfg_wrapper = Config(cfg, epochs=3)
    assert (
        cfg_wrapper.lr == 0.01
        and cfg_wrapper.momentum == 0.95
        and cfg_wrapper.epochs == 3
    )
    with pytest.raises(ValueError):
        Config(3)
