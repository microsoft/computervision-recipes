import os
import pytest
import shutil
import pandas as pd
from pathlib import Path
from utils_ic.datasets import Urls, unzip_url
from utils_ic.experiment import *
from constants import TEMP_DIR


def cleanup_data():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)


@pytest.fixture(scope="module")
def setup_all_datasets(request):
    Experiment.download_benchmark_datasets(TEMP_DIR)
    request.addfinalizer(cleanup_data)


@pytest.fixture(scope="module")
def setup_a_dataset(request):
    os.makedirs(TEMP_DIR)
    unzip_url(Urls.fridge_objects_path, TEMP_DIR, exist_ok=True)
    request.addfinalizer(cleanup_data)


def test_default_experiment_single_dataset(setup_a_dataset):
    """ Test default experiment on a signle dataset. """
    fridge_objects_path = TEMP_DIR / "fridgeObjects"
    exp = Experiment()
    df = exp.run([fridge_objects_path])
    # assert len is 3 because repeat=3 by default
    assert len(df) == 3
    # assert len is 1 because we get the mean on the reps
    assert len(df.mean(level=(1, 2))) == 1
    # assert df is a multi-index dataframe
    assert isinstance(df.index, pd.core.index.MultiIndex)


def test_default_experiment_benchmark_dataset(setup_all_datasets):
    """
    Test default experiment on benchmark dataset.
    WARNING: This test can take a while to execute since we run the experiment
    across all benchmark datasets.
    """
    datasets = [Path(d) for d in os.scandir(TEMP_DIR) if os.path.isdir(d)]
    exp = Experiment()
    df = exp.run(datasets, reps=1)
    # assert len is equal to {len(datasets)}
    assert len(df) == len(datasets)
    # assert df is a multi-index dataframe
    assert isinstance(df.index, pd.core.index.MultiIndex)
