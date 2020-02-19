# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from utils_cv.common.gpu import (
    db_num_workers,
    linux_with_gpu,
    is_binder,
    is_linux,
    is_windows,
    which_processor,
    system_info,
)


def test_which_processor():
    # Naive test: Just run the function to see whether it works or does not work
    which_processor()


def test_is_linux():
    assert type(is_linux()) == bool


def test_is_windows():
    assert type(is_windows()) == bool


def test_linux_with_gpu():
    assert type(linux_with_gpu()) == bool


def test_is_binder():
    assert is_binder() == False


def test_db_num_workers():
    if is_windows():
        assert db_num_workers() == 0
        assert db_num_workers(non_windows_num_workers=7) == 0
    else:
        assert db_num_workers() == 16
        assert db_num_workers(non_windows_num_workers=7) == 7


def test_system_info():
    system_info()
