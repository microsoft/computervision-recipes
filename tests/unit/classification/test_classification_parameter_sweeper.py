# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd
from utils_cv.classification.parameter_sweeper import (
    ParameterSweeper,
    clean_sweeper_df,
    plot_sweeper_df,
)


def _test_sweeper_run(df: pd.DataFrame, df_length: int):
    """ Performs basic tests that all df should pass.
    Args:
        df (pd.DataFame): the df to check
        df_length (int): to assert the len(df) == df_length
    """
    # assert len
    assert len(df) == df_length
    # assert df is a multi-index dataframe
    assert isinstance(df.index, pd.MultiIndex)
    # assert clean_df works
    df = clean_sweeper_df(df)
    assert isinstance(df.index, pd.MultiIndex)
    # assert no error when calling plot_df function
    plot_sweeper_df(df)


def test_default_sweeper_single_dataset(tiny_ic_data_path):
    """ Test default sweeper on a single dataset. """
    sweeper = ParameterSweeper().update_parameters(epochs=[3], im_size=[50])
    df = sweeper.run([tiny_ic_data_path], reps=1)
    _test_sweeper_run(df, df_length=1)

    assert df.mean(level=(1))["accuracy"][0] > 0.0


def test_default_sweeper_benchmark_dataset(tiny_ic_multidata_path):
    """
    Test default sweeper on benchmark dataset.
    WARNING: This test can take a while to execute since we run the sweeper
    across all benchmark datasets.
    """
    sweeper = ParameterSweeper().update_parameters(epochs=[1], im_size=[50])
    df = sweeper.run(tiny_ic_multidata_path, reps=1)
    _test_sweeper_run(df, df_length=len(tiny_ic_multidata_path))

    # assert min accuracy for each dataset
    assert df.mean(level=(2)).loc["fridgeObjectsTiny", "accuracy"] > 0.0
    assert (
        df.mean(level=(2)).loc["fridgeObjectsWatermarkTiny", "accuracy"] > 0.0
    )


def test_update_parameters_01(tiny_ic_data_path):
    """ Tests updating parameters. """
    sweeper = ParameterSweeper()

    # at this point there should only be 1 permutation of the default params
    assert len(sweeper.permutations) == 1
    sweeper.update_parameters(
        learning_rate=[1e-4], im_size=[50, 55], epochs=[1]
    )
    # assert that there are not 6 permutations
    assert len(sweeper.permutations) == 2
    df = sweeper.run([tiny_ic_data_path], reps=1)
    _test_sweeper_run(df, df_length=2)


def test_update_parameters_02():
    """ Tests exception when updating parameters. """
    sweeper = ParameterSweeper()
    with pytest.raises(Exception):
        sweeper.update_parameters(bad_key=[1e-3, 1e-4, 1e-5])
