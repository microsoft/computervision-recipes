# python regular libraries
from base64 import b64encode, b64decode
import json
import os
from typing import Tuple

# fast.ai
from fastai.vision import open_image


def image2bytes(im_list: list, im_dir: str) -> bytes:
    """

    :param im_list: (list of strings) List of image file names
    :param im_dir: (string) Directory name
    :return: List containing the byte arrays of each input image
    """

    string_content_list = []
    for im_name in im_list:
        with open(os.path.join(im_dir, im_name), "rb") as image:
            # Extract image bytes
            im_content = image.read()
            # Convert bytes into a string
            str_im_content = im_content.decode('ISO-8859-1')
            # Append to list of strings
            string_content_list.append(str_im_content)

    concatenated_list = '|||||'.join(string_content_list)
    bytes_of_images = bytes(concatenated_list.encode('ISO-8859-1'))

    return bytes_of_images


def image2json(im_list: list, im_dir: str) -> json:
    """

    :param im_list: (list of strings) List of image file names
    :param im_dir: (string) Directory name
    :return: List containing the byte arrays of each input image
    """

    im_string_list = []
    for im_name in im_list:
        with open(os.path.join(im_dir, im_name), "rb") as image:
            # Extract image bytes
            im_content = image.read()
            # Convert bytes into a string
            im_string_list.append(b64encode(im_content).decode('utf-8'))

    input_to_service = json.dumps({'data': im_string_list})

    return input_to_service
