import os
import numpy as np
from pathlib import Path
from PIL import Image
import pytest
from utils_ic.common import data_path, get_files_in_directory, ic_root_path, im_height, im_width, im_width_height



def test_ic_root_path():
    s = ic_root_path()
    assert isinstance(s, str) and s != ""


def test_data_path():
    s = data_path()
    assert isinstance(s, str) and s != ""


def test_im_width(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path)/"can"/"1.jpg"
    assert (
        im_width(im_path) == 499
    ), "Expected image width of 499, but got {}".format(im_width(im_path))
    im = np.zeros((100, 50))
    assert im_width(im) == 50, "Expected image width of 50, but got ".format(
        im_width(im)
    )


def test_im_height(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path)/"can"/"1.jpg"
    assert (
        im_height(im_path) == 665
    ), "Expected image height of 665, but got ".format(im_width(60))
    im = np.zeros((100, 50))
    assert (
        im_height(im) == 100
    ), "Expected image height of 100, but got ".format(im_width(im))


def test_im_width_height(tiny_ic_data_path):
    im_path = Path(tiny_ic_data_path)/"can"/"1.jpg"
    w, h = im_width_height(im_path)
    assert w == 499 and h == 665
    im = np.zeros((100, 50))
    w, h = im_width_height(im)
    assert w == 50 and h == 100


def test_get_files_in_directory(tiny_ic_data_path):
    im_dir = os.path.join(tiny_ic_data_path, "can")
    assert len(get_files_in_directory(im_dir)) == 22
    assert len(get_files_in_directory(im_dir, suffixes=[".jpg"])) == 22
    assert len(get_files_in_directory(im_dir, suffixes=[".nonsense"])) == 0
