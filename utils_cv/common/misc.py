# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
import random
import shutil
from typing import List, Union
from PIL import ImageFont
from typing import Any

import numpy as np


def set_random_seed(s: int):
    """Set random seed
    """
    np.random.seed(s)
    random.seed(s)

    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(s)
            torch.cuda.manual_seed_all(s)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def copy_files(
    fpaths: Union[str, List[str]],
    dst_dir: str,
    infer_subdir: bool = False,
    remove: bool = False,
):
    """Copy list of files into destination

    Args:
        fpaths: File path to copy
        dst_dir: Destination directory
        infer_subdir: If True, try to infer directory structure of the files and copy.
            Otherwise, just copy the files to dst
        remove: Remove copied files from the original directory
    """
    if isinstance(fpaths, (str, Path)):
        fpaths = [fpaths]

    for fpath in fpaths:
        if infer_subdir:
            dst = os.path.join(
                dst_dir, os.path.basename(os.path.dirname(fpath))
            )
        else:
            dst = dst_dir

        if not os.path.isdir(dst):
            os.makedirs(dst)
        shutil.copy(fpath, dst)

        if remove:
            os.remove(fpath)


def get_font(size: int = 12) -> ImageFont:
    """ Gets a font object. 
        Tries different fonts and lower/upper case to be compatible with both Linux and Windows.
    """
    font = None
    for (
        font_name
    ) in "Tahoma tahoma Verdana verdana Arial arial Helvetica helvetica DejaVuSans dejavusans".split():
        try:
            font = ImageFont.truetype(f"{font_name}.ttf", size)
        except (AttributeError, IOError):
            font = None
        if font:
            break
    else:
        try:
            font = ImageFont.load_default()
        except (AttributeError, IOError):
            font = None

    return font


class Config(object):
    def __init__(self, config=None, **extras):
        """Dictionary wrapper to access keys as attributes.
        Args:
            config (dict or Config): Configurations
            extras (kwargs): Extra configurations
        Examples:
            >>> cfg = Config({'lr': 0.01}, momentum=0.95)
            or
            >>> cfg = Config({'lr': 0.01, 'momentum': 0.95})
            then, use as follows:
            >>> print(cfg.lr, cfg.momentum)
        """
        if config is not None:
            if isinstance(config, dict):
                for k in config:
                    setattr(self, k, config[k])
            elif isinstance(config, self.__class__):
                self.__dict__ = config.__dict__.copy()
            else:
                raise ValueError("Unknown config")

        for k, v in extras.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        setattr(self, key, value)
