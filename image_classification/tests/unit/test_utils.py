import os
import pytest
import shutil
import unittest
from pathlib import Path
from typing import Union
from utils_ic.datasets import Urls, unzip_url


class TestUnzipUrl(unittest.TestCase):
    """
    This class tests the unzip_url function
    """

    TEMP_DIR = "../tmp_data"

    def setUp(self):
        os.makedirs(self.TEMP_DIR, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.TEMP_DIR):
            shutil.rmtree(self.TEMP_DIR)

    def _test_url_data(self, url: str, path: Union[Path, str], dir_name: str):
        data_path = unzip_url(url, fpath=path, dest=path, overwrite=True)
        # assert zip file exists
        self.assertTrue(os.path.exists(os.path.join(path, f"{dir_name}.zip")))
        # assert unzipped file (titled {dir_name}) exists
        self.assertTrue(os.path.exists(os.path.join(path, dir_name)))
        # assert unzipped file equals the returned {data_path}
        self.assertEqual(
            os.path.realpath(os.path.join(path, dir_name)), os.path.realpath(data_path)
        )

    def test_unzip_url_rel_path(self):
        """ Test unzip with relative path. """
        rel_path = Path(self.TEMP_DIR)
        self._test_url_data(Urls.lettuce_path, rel_path, "lettuce")
        self._test_url_data(Urls.fridge_objects_path, rel_path, "fridgeObjects")
        self._test_url_data(Urls.recycle_path, rel_path, "recycle_v3")

    def test_unzip_url_abs_path(self):
        """ Test unzip with absolute path. """
        abs_path = Path(os.path.abspath(self.TEMP_DIR))
        self._test_url_data(Urls.lettuce_path, abs_path, "lettuce")
        self._test_url_data(Urls.fridge_objects_path, abs_path, "fridgeObjects")
        self._test_url_data(Urls.recycle_path, abs_path, "recycle_v3")
