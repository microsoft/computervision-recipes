# python regular libraries
import json
import os

from base64 import b64encode


def im2base64(im_name: str, im_dir: str) -> bytes:
    """

    Args:
        im_name (string): Image file name
        im_dir (string): Image directory name

    Returns: im_bytes

    """

    with open(os.path.join(im_dir, im_name), "rb") as image:
        # Extract image bytes
        im_content = image.read()
        # Convert bytes into a string
        im_bytes = b64encode(im_content)

    return im_bytes


def ims2json(im_list: list, im_dir: str) -> json:
    """

    Args:
        im_list (list of strings): List of image file names
        im_dir (string): Directory name

    Returns: input_to_service: String containing the based64-encoded images
    decoded into strings

    """

    im_string_list = []
    for im_name in im_list:
        im_string_list.append(im2base64(im_name, im_dir).decode('utf-8'))

    input_to_service = json.dumps({'data': im_string_list})

    return input_to_service
