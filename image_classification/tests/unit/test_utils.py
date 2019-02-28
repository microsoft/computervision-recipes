import os
import pytest
import shutil
from pathlib import Path
from typing import Union
from utils_ic.datasets import Urls, unzip_url

# temporarily putting this constant here until we add a way to manage constants in tests
TEMP_DIR = "../tmp_data"


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
    rel_path = Path(TEMP_DIR)
    _test_url_data(Urls.lettuce_path, rel_path, "lettuce")
    _test_url_data(Urls.fridge_objects_path, rel_path, "fridgeObjects")
    _test_url_data(Urls.recycle_path, rel_path, "recycle_v3")


def test_unzip_url_abs_path(make_temp_data_dir):
    """ Test unzip with absolute path. """
    abs_path = Path(os.path.abspath(TEMP_DIR))
    _test_url_data(Urls.lettuce_path, abs_path, "lettuce")
    _test_url_data(Urls.fridge_objects_path, abs_path, "fridgeObjects")
    _test_url_data(Urls.recycle_path, abs_path, "recycle_v3")
