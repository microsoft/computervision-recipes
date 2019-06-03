# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import shutil
from pathlib import Path
from typing import Union
from utils_cv.classification.data import Urls
from utils_cv.common.data import (
    data_path,
    get_files_in_directory,
    unzip_url,
    root_path,
)


def test_root_path():
    s = root_path()
    assert isinstance(s, Path) and s != ""


def test_data_path():
    s = data_path()
    assert isinstance(s, Path) and s != ""


def test_get_files_in_directory(tiny_ic_data_path):
    im_dir = os.path.join(tiny_ic_data_path, "can")
    assert len(get_files_in_directory(im_dir)) == 22
    assert len(get_files_in_directory(im_dir, suffixes=[".jpg"])) == 22
    assert len(get_files_in_directory(im_dir, suffixes=[".nonsense"])) == 0


def _test_url_data(url: str, path: Union[Path, str], dir_name: str):
    data_path = unzip_url(url, fpath=path, dest=path, exist_ok=True)
    # assert zip file exists
    assert os.path.exists(os.path.join(path, f"{dir_name}.zip"))
    # assert unzipped file (titled {dir_name}) exists
    assert os.path.exists(os.path.join(path, dir_name))
    # assert unzipped file equals the returned {data_path}
    assert os.path.realpath(os.path.join(path, dir_name)) == os.path.realpath(
        data_path
    )


def test_unzip_url_rel_path(tmp_path):
    """ Test unzip with relative path. """
    rel_path = tmp_path
    _test_url_data(Urls.fridge_objects_path, rel_path, "fridgeObjects")


def test_unzip_url_abs_path(tmp_path):
    """ Test unzip with absolute path. """
    abs_path = Path(os.path.abspath(tmp_path))
    _test_url_data(Urls.fridge_objects_path, abs_path, "fridgeObjects")


def test_unzip_url_exist_ok(tmp_path):
    """
    Test if exist_ok is true and (file exists, file does not exist)
    """
    os.makedirs(tmp_path / "fridgeObjects")
    fridge_object_path = unzip_url(
        Urls.fridge_objects_path, tmp_path, exist_ok=True
    )
    assert len(os.listdir(fridge_object_path)) == 0
    shutil.rmtree(tmp_path / "fridgeObjects")
    fridge_object_path = unzip_url(
        Urls.fridge_objects_watermark_path, tmp_path, exist_ok=True
    )
    assert len(os.listdir(fridge_object_path)) > 0


def test_unzip_url_not_exist_ok(tmp_path):
    """
    Test if exist_ok is false and (file exists, file does not exist)
    """
    os.makedirs(tmp_path / "fridgeObjects")
    with pytest.raises(FileExistsError):
        unzip_url(Urls.fridge_objects_path, tmp_path, exist_ok=False)
    shutil.rmtree(tmp_path / "fridgeObjects")
    os.remove(tmp_path / "fridgeObjects.zip")
    unzip_url(Urls.fridge_objects_path, tmp_path, exist_ok=False)
