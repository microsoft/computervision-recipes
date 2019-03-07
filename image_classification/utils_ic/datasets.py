import os
import requests
import shutil
from pathlib import Path
from typing import List, Union
from urllib.parse import urljoin, urlparse
from zipfile import ZipFile

Url = str


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp.blob.core.windows.net/public/datasets/image_classification/"

    # Same link Keras is using
    imagenet_labels_json = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"

    # datasets
    fridge_objects_path = urljoin(base, "fridgeObjects.zip")
    food_101_subset_path = urljoin(base, "food101Subset.zip")
    flickr_logos_32_subset_path = urljoin(base, "flickrLogos32Subset.zip")
    lettuce_path = urljoin(base, "lettuce.zip")
    recycle_path = urljoin(base, "recycle_v3.zip")

    @classmethod
    def all(cls) -> List[Url]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]


def imagenet_labels() -> list:
    """List of ImageNet labels with the original index.

    Returns:
         list: ImageNet labels
    """
    labels = requests.get(Urls.imagenet_labels_json).json()
    return [labels[str(k)][1] for k in range(len(labels))]


def data_path() -> Path:
    """Get the data path"""
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data")
    )


def _get_file_name(url: str) -> str:
    """Get a file name based on url"""
    return urlparse(url).path.split("/")[-1]


def unzip_url(
    url: str,
    fpath: Union[Path, str] = data_path(),
    dest: Union[Path, str] = data_path(),
    exist_ok: bool = False,
    overwrite: bool = False,
) -> Path:
    """
    Download file from URL to {fpath} and unzip to {dest}.
    {fpath} and {dest} must be directories
    Params:
        exist_ok: if exist_ok, then skip if exists, otherwise throw error
        overwrite: if overwrite, remove zipped file and unziped dir
    Returns path of {dest}
    """
    assert os.path.exists(fpath)
    assert os.path.exists(dest)

    fname = _get_file_name(url)
    fname_without_extension = fname.split(".")[0]
    zip_file = Path(os.path.join(fpath, fname))
    unzipped_dir = Path(os.path.join(fpath, fname_without_extension))

    if overwrite:
        try:
            os.remove(zip_file)
        except OSError as e:
            pass
        try:
            shutil.rmtree(unzipped_dir)
        except OSError as e:
            pass

    try:
        # download zipfile if zipfile not exists
        if zip_file.is_file():
            raise FileExistsError(zip_file)
        else:
            r = requests.get(url)
            f = open(zip_file, "wb")
            f.write(r.content)
            f.close()

        # unzip downloaded zipfile if dir not exists
        if unzipped_dir.is_dir():
            raise FileExistsError(unzipped_dir)
        else:
            z = ZipFile(zip_file, "r")
            z.extractall(fpath)
            z.close()
    except FileExistsError:
        if not exist_ok:
            print("File already exists. Use param {exist_ok} to ignore.")
            raise

    return os.path.realpath(os.path.join(fpath, fname_without_extension))
