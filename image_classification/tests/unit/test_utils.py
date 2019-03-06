import os
import pytest
import shutil
from pathlib import Path
from typing import Union
from utils_ic.datasets import Urls, unzip_url, imagenet_labels

# temporarily putting this constant here until we add a way to manage constants in tests
TEMP_DIR = Path("../tmp_data")


@pytest.fixture(scope="function")
def make_temp_data_dir(request):
    os.makedirs(TEMP_DIR, exist_ok=True)

    def remove_temp_data_dir():
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)

    request.addfinalizer(remove_temp_data_dir)


def _test_url_data(url: str, path: Union[Path, str], dir_name: str):
    data_path = unzip_url(url, fpath=path, dest=path, overwrite=True)
    # assert zip file exists
    assert os.path.exists(os.path.join(path, f"{dir_name}.zip"))
    # assert unzipped file (titled {dir_name}) exists
    assert os.path.exists(os.path.join(path, dir_name))
    # assert unzipped file equals the returned {data_path}
    assert os.path.realpath(os.path.join(path, dir_name)) == os.path.realpath(
        data_path
    )


def test_unzip_url_rel_path(make_temp_data_dir):
    """ Test unzip with relative path. """
    rel_path = TEMP_DIR
    _test_url_data(Urls.lettuce_path, rel_path, "lettuce")
    _test_url_data(Urls.fridge_objects_path, rel_path, "fridgeObjects")
    _test_url_data(Urls.recycle_path, rel_path, "recycle_v3")


def test_unzip_url_abs_path(make_temp_data_dir):
    """ Test unzip with absolute path. """
    abs_path = Path(os.path.abspath(TEMP_DIR))
    _test_url_data(Urls.lettuce_path, abs_path, "lettuce")
    _test_url_data(Urls.fridge_objects_path, abs_path, "fridgeObjects")
    _test_url_data(Urls.recycle_path, abs_path, "recycle_v3")


def text_unzip_url_overwrite(make_temp_data_dir):
    """ Test if overwrite is true and file exists """
    os.makedirs(TEMP_DIR/"fridgeObjects")
    fridge_objects_path = unzip_url(Urls.fridge_objects_path, TEMP_DIR, overwrite=True)
    assert len(os.listdir(fridge_objects_path)) >= 0
    os.makedirs(TEMP_DIR/"lettuce")
    lettuce_path = unzip_url(Urls.lettuce_path, TEMP_DIR, overwrite=False)
    assert len(os.listdir(lettuce_path)) == 0


def test_unzip_url_exist_ok(make_temp_data_dir):
    """
    Test if exist_ok is true and (file exists, file does not exist)
    """
    os.makedirs(TEMP_DIR/"recycle_v3")
    recycle_path = unzip_url(Urls.recycle_path, TEMP_DIR, exist_ok=True)
    assert len(os.listdir(recycle_path)) == 0
    lettuce_path = unzip_url(Urls.lettuce_path, TEMP_DIR, exist_ok=True)
    assert len(os.listdir(lettuce_path)) >= 0


def test_unzip_url_not_exist_ok(make_temp_data_dir):
    """
    Test if exist_ok is false and (file exists, file does not exist)
    """
    os.makedirs(TEMP_DIR/"fridgeObjects")
    with pytest.raises(FileExistsError):
        unzip_url(Urls.fridge_objects_path, TEMP_DIR, exist_ok=False)

    open(TEMP_DIR/"lettuce.zip", 'a').close()
    with pytest.raises(FileExistsError):
        unzip_url(Urls.lettuce_path, TEMP_DIR, exist_ok=False)


def test_imagenet_labels():
    # Compare first five labels for quick check
    IMAGENET_LABELS_FIRST_FIVE = ("tench", "goldfish", "great_white_shark", "tiger_shark", "hammerhead")

    labels = imagenet_labels()
    for i in range(5):
        assert labels[i] == IMAGENET_LABELS_FIRST_FIVE[i]
