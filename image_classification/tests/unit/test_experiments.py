import os
import pytest
import shutil

# import pandas as pd
# from pathlib import Path
from utils_ic.datasets import Urls, unzip_url
from utils_ic.parameter_sweeper import *
from constants import TEMP_DIR


def cleanup_data():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


@pytest.fixture(scope="module")
def setup_all_datasets(request):
    """ Sets up all available datasets for testing on. """
    ParameterSweeper.download_benchmark_datasets(TEMP_DIR)
    request.addfinalizer(cleanup_data)


@pytest.fixture(scope="module")
def setup_a_dataset(request):
    """ Sets up a dataset for testing on. """
    os.makedirs(TEMP_DIR)
    unzip_url(Urls.fridge_objects_path, TEMP_DIR, exist_ok=True)
    request.addfinalizer(cleanup_data)


# def _test_sweeper_run(df: pd.DataFrame, df_length: int):
#     """ Performs basic tests that all df should pass.
#     Args:
#         df (pd.DataFame): the df to check
#         df_length (int): to assert the len(df) == df_length
#     """
#     # assert len
#     assert len(df) == df_length
#     # assert df is a multi-index dataframe
#     assert isinstance(df.index, pd.core.index.MultiIndex)
#     # assert clean_df works
#     df = clean_df(df)
#     assert isinstance(df.index, pd.core.index.MultiIndex)
#     # assert no error when calling plot_df function
#     plot_df(df)
#
#
# def test_default_sweeper_single_dataset(setup_a_dataset):
#     """ Test default sweeper on a single dataset. """
#     fridge_objects_path = TEMP_DIR / "fridgeObjects"
#     sweeper = ParameterSweeper()
#     df = sweeper.run([fridge_objects_path])
#     _test_sweeper_run(df, df_length=3)
#
#     # assert accuracy over 3 runs is > 85%
#     assert df.mean(level=(1))["accuracy"][0] > 0.85
#
#
# def test_default_sweeper_benchmark_dataset(setup_all_datasets):
#     """
#     Test default sweeper on benchmark dataset.
#     WARNING: This test can take a while to execute since we run the sweeper
#     across all benchmark datasets.
#     """
#     datasets = [Path(d) for d in os.scandir(TEMP_DIR) if os.path.isdir(d)]
#     sweeper = ParameterSweeper()
#     df = sweeper.run(datasets, reps=1)
#     _test_sweeper_run(df, df_length=len(datasets))
#
#     # assert min accuracy for each dataset
#     assert df.mean(level=(2)).loc["fridgeObjects", "accuracy"] > 0.85
#     assert df.mean(level=(2)).loc["food101Subset", "accuracy"] > 0.75
#     assert df.mean(level=(2)).loc["fashionTexture", "accuracy"] > 0.70
#     assert df.mean(level=(2)).loc["flickrLogos32Subset", "accuracy"] > 0.75
#     assert df.mean(level=(2)).loc["lettuce", "accuracy"] > 0.70
#     assert df.mean(level=(2)).loc["recycle_v3", "accuracy"] > 0.85
#
#
# def test_update_parameters_01(setup_a_dataset):
#     """ Tests updating parameters. """
#     fridge_objects_path = TEMP_DIR / "fridgeObjects"
#     sweeper = ParameterSweeper()
#
#     # at this point there should only be 1 permutation of the default params
#     assert len(sweeper.permutations) == 1
#     sweeper.update_parameters(
#         learning_rate=[1e-3, 1e-4, 1e-5], im_size=[299, 499], epochs=[5]
#     )
#     # assert that there are not 6 permutations
#     assert len(sweeper.permutations) == 6
#     df = sweeper.run([fridge_objects_path])
#     _test_sweeper_run(df, df_length=18)
#
#
# def test_update_parameters_02(setup_a_dataset):
#     """ Tests exception when updating parameters. """
#     sweeper = ParameterSweeper()
#     with pytest.raises(Exception):
#         sweeper.update_parameters(bad_key=[1e-3, 1e-4, 1e-5])
