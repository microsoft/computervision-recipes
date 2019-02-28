import os
import requests
import shutil
from pathlib import Path
from typing import Union
from urllib.parse import urlparse, urljoin
from zipfile import ZipFile


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp.blob.core.windows.net/public/datasets/image_classification/"

    # datasets
    fridge_objects_path = urljoin(base, "fridgeObjects.zip")
    food_101_subset_path = urljoin(base, "food101Subset.zip")
    flickr_logos_32_subset_path = urljoin(base, "flickrLogos32Subset.zip")
    lettuce_path = urljoin(base, "lettuce.zip")
    recycle_path = urljoin(base, "recycle_v3.zip")


def data_path() -> Path:
    """Get the data path"""
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir, "data"))


def _get_file_name(url: str) -> str:
    """Get a file name based on url"""
    return urlparse(url).path.split("/")[-1]


def unzip_url(
    url: str,
    fpath: Union[Path, str] = data_path(),
    dest: Union[Path, str] = data_path(),
    overwrite: bool = False,
) -> Path:
    """
    Download file from URL to {fpath} and unzip to {dest}. 
    {fpath} and {dest} must be directories
    Returns path of {dest}
    """
    assert os.path.exists(fpath)
    assert os.path.exists(dest)

    fname = _get_file_name(url)
    if os.path.exists(os.path.join(fpath, fname)):
        if overwrite:
            os.remove(os.path.join(fpath, fname))
        else:
            raise Exception(f"{fname} already exists in {fpath}.")

    fname_without_extension = fname.split(".")[0]
    if os.path.exists(os.path.join(fpath, fname_without_extension)):
        if overwrite:
            shutil.rmtree(os.path.join(fpath, fname_without_extension))
        else:
            raise Exception(f"{fname_without_extension} already exists in {fpath}.")

    r = requests.get(url)
    f = open(os.path.join(fpath, fname), "wb")
    f.write(r.content)
    f.close()

    os.makedirs(os.path.join(fpath, fname_without_extension))
    z = ZipFile(os.path.join(fpath, fname), "r")
    z.extractall(os.path.join(fpath))
    z.close()

    return os.path.realpath(os.path.join(fpath, fname_without_extension))
