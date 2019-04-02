import pytest
import pandas as pd
from utils_ic.parameter_sweeper import *


def _test_sweeper_run(df: pd.DataFrame, df_length: int):
    """ Performs basic tests that all df should pass.
    Args:
        df (pd.DataFame): the df to check
        df_length (int): to assert the len(df) == df_length
    """
    # assert len
    assert len(df) == df_length
    # assert df is a multi-index dataframe
    assert isinstance(df.index, pd.core.index.MultiIndex)
    # assert clean_df works
    df = clean_sweeper_df(df)
    assert isinstance(df.index, pd.core.index.MultiIndex)
    # assert no error when calling plot_df function
    plot_sweeper_df(df)


def test_default_sweeper_single_dataset(dataset):
    """ Test default sweeper on a single dataset. """
    sweeper = ParameterSweeper().update_parameters(epochs=[5])
    df = sweeper.run([dataset], reps=1)
    _test_sweeper_run(df, df_length=1)

    # assert accuracy over 3 runs is > 85%
    assert df.mean(level=(1))["accuracy"][0] > 0.0


def test_default_sweeper_benchmark_dataset(multidataset):
    """
    Test default sweeper on benchmark dataset.
    WARNING: This test can take a while to execute since we run the sweeper
    across all benchmark datasets.
    """
    sweeper = ParameterSweeper().update_parameters(epochs=[5])
    df = sweeper.run(multidataset, reps=1)
    _test_sweeper_run(df, df_length=len(multidataset))

    # assert min accuracy for each dataset
    assert df.mean(level=(2)).loc["fridgeObjectsTiny", "accuracy"] > 0.0
    assert (
        df.mean(level=(2)).loc["fridgeObjectsWatermarkTiny", "accuracy"] > 0.0
    )


def test_update_parameters_01(dataset):
    """ Tests updating parameters. """
    sweeper = ParameterSweeper()

    # at this point there should only be 1 permutation of the default params
    assert len(sweeper.permutations) == 1
    sweeper.update_parameters(
        learning_rate=[1e-3, 1e-4, 1e-5], im_size=[299, 499], epochs=[5]
    )
    # assert that there are not 6 permutations
    assert len(sweeper.permutations) == 6
    df = sweeper.run([dataset], reps=1)
    _test_sweeper_run(df, df_length=6)


def test_update_parameters_02():
    """ Tests exception when updating parameters. """
    sweeper = ParameterSweeper()
    with pytest.raises(Exception):
        sweeper.update_parameters(bad_key=[1e-3, 1e-4, 1e-5])
