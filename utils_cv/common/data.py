# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
import requests
from typing import List, Union
from urllib.parse import urlparse
from zipfile import ZipFile


def data_path() -> Path:
    """Get the data directory path"""
    data_dir = Path(
        os.path.realpath(
            os.path.join(
                os.path.dirname(__file__), os.pardir, os.pardir, "data"
            )
        )
    )
    data_dir.mkdir(exist_ok=True)
    return data_dir


def download(url: str, loc: str):
    """ Download contents of a url into 'loc'"""
    r = requests.get(url)
    with open(loc, 'wb') as f:
        f.write(r.content)
    return loc


def get_files_in_directory(
    directory: str, suffixes: List[str] = None
) -> List[str]:
    """Returns all filenames in a directory which optionally match one of multiple suffixes.
    Args:
        directory: directory to scan for files.
        suffixes: only keep the filenames which ends with one of the suffixes
            (e.g. suffixes = [".jpg", ".png", ".gif"]).
    Return:
        List of filenames
    """
    if not os.path.exists(directory):
        raise Exception(f"Directory '{directory}' does not exist.")
    filenames = [str(p) for p in Path(directory).iterdir() if p.is_file()]
    if suffixes and suffixes != "":
        filenames = [
            s for s in filenames if s.lower().endswith(tuple(suffixes))
        ]
    return sorted(filenames)


def _get_file_name(url: str) -> str:
    """ Get a file name based on url. """
    return urlparse(url).path.split("/")[-1]


def unzip_url(
    url: str,
    fpath: Union[Path, str] = None,
    dest: Union[Path, str] = None,
    exist_ok: bool = False,
) -> str:
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

    if fpath is None and dest is None:
        fpath = data_path()
        dest = data_path()
    if fpath is None and dest is not None:
        fpath = dest
    if dest is None and fpath is not None:
        dest = fpath

    os.makedirs(dest, exist_ok=True)
    os.makedirs(fpath, exist_ok=True)

    fname = _get_file_name(url)
    fname_without_extension = fname.split(".")[0]
    zip_file = Path(os.path.join(fpath, fname))
    unzipped_dir = Path(os.path.join(dest, fname_without_extension))

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

    return os.path.realpath(unzipped_dir)


def unzip_urls(
    urls: List[str], dest: Union[Path, str] = data_path()
) -> List[str]:
    """ Download and unzip all datasets in Urls to dest """

    # make dir if not exist
    if not Path(dest).is_dir():
        os.makedirs(dest)

    # download all data urls
    paths = list()
    for url in urls:
        paths.append(unzip_url(url, fpath=dest, dest=dest, exist_ok=True))

    return paths


def root_path() -> Path:
    """Get path of root dir."""
    return Path(
        os.path.realpath(
            os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
        )
    )
