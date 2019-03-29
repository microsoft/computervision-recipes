import os
import requests
from .common import data_path
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
    fridge_objects_watermark_path = urljoin(base, "fridgeObjectsWatermark.zip")
    fridge_objects_tiny_path = urljoin(base, "fridgeObjectsTiny.zip")
    fridge_objects_watermark_tiny_path = urljoin(
        base, "fridgeObjectsWatermarkTiny.zip"
    )
    food_101_subset_path = urljoin(base, "food101Subset.zip")
    fashion_texture_path = urljoin(base, "fashionTexture.zip")
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


def _get_file_name(url: str) -> str:
    """ Get a file name based on url. """
    return urlparse(url).path.split("/")[-1]


def unzip_url(
    url: str,
    fpath: Union[Path, str] = data_path(),
    dest: Union[Path, str] = data_path(),
    exist_ok: bool = False,
) -> Path:
    """ Download file from URL to {fpath} and unzip to {dest}.
    {fpath} and {dest} must be directories

    Args:
        url (str): url to download from
        fpath (Union[Path, str]): The location to save the url zip file to
        dest (Union[Path, str]): The destination to unzip {fpath}
        exist_ok (bool): if exist_ok, then skip if exists, otherwise throw error

    Raises:
        FileExistsError: if file exists

    Returns:
        Path of {dest}
    """

    def _raise_file_exists_error(path: Union[Path, str]) -> None:
        if not exist_ok:
            raise FileExistsError(path, "Use param {{exist_ok}} to ignore.")

    assert os.path.exists(fpath)
    assert os.path.exists(dest)

    fname = _get_file_name(url)
    fname_without_extension = fname.split(".")[0]
    zip_file = Path(os.path.join(fpath, fname))
    unzipped_dir = Path(os.path.join(fpath, fname_without_extension))

    # download zipfile if zipfile not exists
    if zip_file.is_file():
        _raise_file_exists_error(zip_file)
    else:
        r = requests.get(url)
        f = open(zip_file, "wb")
        f.write(r.content)
        f.close()

    # unzip downloaded zipfile if dir not exists
    if unzipped_dir.is_dir():
        _raise_file_exists_error(unzipped_dir)
    else:
        z = ZipFile(zip_file, "r")
        z.extractall(fpath)
        z.close()

    return os.path.realpath(os.path.join(fpath, fname_without_extension))


def unzip_urls(
    urls: List[Url], dest: Union[Path, str] = data_path()
) -> List[Path]:
    """ Download and unzip all datasets in Urls to dest """

    # make dir if not exist
    if not Path(dest).is_dir():
        os.makedirs(dest)

    # download all data urls
    paths = list()
    for url in urls:
        paths.append(unzip_url(url, dest, exist_ok=True))

    return paths
