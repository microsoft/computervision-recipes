import os
import numpy as np
from PIL import Image
import pytest
from constants import TEMP_DIR
from utils_ic.common import *


# Image path used for unit tests
im_path = os.path.join(TEMP_DIR, "example.jpg")


def cleanup_data():
    """ Removes data after testing. """
    filesToRemove = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR)]
    for f in filesToRemove:
        os.remove(f)
    os.rmdir(TEMP_DIR)


@pytest.fixture(scope="module")
def setup_all_data(request):
    """ Sets up data for testing. """
    os.makedirs(TEMP_DIR, exist_ok=True)
    Image.new("RGB", (60, 30), color="red").save(im_path)
    request.addfinalizer(cleanup_data)


def test_ic_root_path():
    s = ic_root_path()
    assert isinstance(s, str) and s != ""


def test_data_path():
    s = data_path()
    assert isinstance(s, str) and s != ""


def test_im_width(setup_all_data):
    assert (
        im_width(im_path) == 60
    ), "Expected image width of 60, but got {}".format(im_width(im_path))
    im = np.zeros((100, 50))
    assert im_width(im) == 50, "Expected image width of 50, but got ".format(
        im_width(im)
    )


def test_im_height(setup_all_data):
    assert (
        im_height(im_path) == 30
    ), "Expected image height of 30, but got ".format(im_width(60))
    im = np.zeros((100, 50))
    assert (
        im_height(im) == 100
    ), "Expected image height of 100, but got ".format(im_width(im))


def test_im_width_height(setup_all_data):
    w, h = im_width_height(im_path)
    assert w == 60 and h == 30
    im = np.zeros((100, 50))
    w, h = im_width_height(im)
    assert w == 50 and h == 100


def test_get_files_in_directory(setup_all_data):
    assert len(get_files_in_directory(TEMP_DIR)) == 1
    assert len(get_files_in_directory(TEMP_DIR, suffixes=[".jpg"])) == 1
    assert len(get_files_in_directory(TEMP_DIR, suffixes=[".nonsense"])) == 0
    open(os.path.join(TEMP_DIR, "new.jpg"), "w").close()
    assert len(get_files_in_directory(TEMP_DIR, suffixes=[".jpg"])) == 2
