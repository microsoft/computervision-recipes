import os
from pathlib import Path


def ic_root_path() -> Path:
    """Get the image classification root path"""
    return os.path.realpath(os.path.join(os.path.dirname(__file__), os.pardir))


def data_path() -> Path:
    """Get the data directory path"""
    return os.path.realpath(
        os.path.join(os.path.dirname(__file__), os.pardir, "data")
    )
