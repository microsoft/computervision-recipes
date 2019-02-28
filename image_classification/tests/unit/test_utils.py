import pytest
from ic_utils.datasets import Urls, unzip_url
import os
from pathlib import Path
import unittest
import shutil


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

    def test_unzip_url_rel_path(self):
        """ Test unzip with relative path. """
        rel_path = Path(self.TEMP_DIR)
        data_path = unzip_url(
            Urls.lettuce, fpath=rel_path, dest=rel_path, overwrite=True
        )
        self.assertTrue(os.path.exists(os.path.join(rel_path, "lettuce.zip")))
        self.assertTrue(os.path.exists(os.path.join(rel_path, "lettuce")))
        self.assertEqual(
            os.path.realpath(os.path.join(rel_path, "lettuce")),
            os.path.realpath(data_path),
        )

    def test_unzip_url_abs_path(self):
        """ Test unzip with absolute path. """
        abs_path = Path(os.path.abspath(self.TEMP_DIR))
        data_path = unzip_url(
            Urls.lettuce, fpath=abs_path, dest=abs_path, overwrite=True
        )
        self.assertTrue(os.path.exists(os.path.join(abs_path, "lettuce.zip")))
        self.assertTrue(os.path.exists(os.path.join(abs_path, "lettuce")))
        self.assertEqual(
            os.path.realpath(os.path.join(abs_path, "lettuce")),
            os.path.realpath(data_path),
        )
