import os
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Union, Tuple, List


def ic_root_path() -> Path:
    """Get the image classification root path"""
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))


def data_path() -> Path:
    """Get the data directory path"""
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data")
    )


def im_width(input: Union[str, np.array]) -> int:
    """Returns the width of an image.
    Args:
        input: Image path or image as numpy array.
    Return:
        Image width.
    """
    return im_width_height(input)[0]


def im_height(input: Union[str, np.array]) -> int:
    """Returns the height of an image.
    Args:
        input: Image path or image as numpy array.
    Return:
        Image height.
    """
    return im_width_height(input)[1]


def im_width_height(input: Union[str, np.array]) -> Tuple[int, int]:
    """Returns the width and height of an image.
    Args:
        input: Image path or image as numpy array.
    Return:
        Tuple of ints (width,height).
    """
    if isinstance(input, str) or isinstance(input, Path):
        width, height = Image.open(
            input
        ).size  # this is fast since it does not load the full image
    else:
        width, height = (input.shape[1], input.shape[0])
    return width, height


def get_files_in_directory(
    directory: str, suffixes: List[str] = None
) -> List[str]:
    """Returns all filenames in a directory which optionally match one of multiple suffixes.
    Args:
        directory: directory to scan for files.
        suffixes: only keep the filenames which ends with one of the suffixes (e.g. suffixes = [".jpg", ".png", ".gif"]).
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
    return filenames
