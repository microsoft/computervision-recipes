# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# python regular libraries
from pathlib import Path
from typing import Union

from base64 import b64encode


def im2base64(im_path: Union[Path, str]) -> bytes:
    """

    Args:
        im_path (string): Path to the image

    Returns: im_bytes

    """

    with open(im_path, "rb") as image:
        # Extract image bytes
        im_content = image.read()
        # Convert bytes into a string
        im_bytes = b64encode(im_content)

    return im_bytes


def ims2strlist(im_path_list: list) -> list:
    """

    Args:
        im_path_list (list of strings): List of image paths

    Returns: im_string_list: List containing based64-encoded images
    decoded into strings

    """

    im_string_list = []
    for im_path in im_path_list:
        im_string_list.append(im2base64(im_path).decode("utf-8"))

    return im_string_list
