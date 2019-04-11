import os
import pytest
import shutil
from pathlib import Path
from typing import Union

from utils_ic.datasets import Urls, unzip_url, imagenet_labels


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
    fridge_object_path = unzip_url(Urls.recycle_path, tmp_path, exist_ok=True)
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


def test_imagenet_labels():
    # Compare first five labels for quick check
    IMAGENET_LABELS_FIRST_FIVE = (
        "tench",
        "goldfish",
        "great_white_shark",
        "tiger_shark",
        "hammerhead",
    )

    labels = imagenet_labels()
    for i in range(5):
        assert labels[i] == IMAGENET_LABELS_FIRST_FIVE[i]
