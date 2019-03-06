# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests


import os
import papermill as pm
import shutil
from utils_ic.datasets import Urls, unzip_url
from tests.conftest import path_notebooks

# Unless manually modified, python3 should be the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


def test_webcam_notebook_run(notebooks):
    notebook_path = notebooks["00_webcam"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )


def test_simple_notebook_run(notebooks):
    notebook_path = notebooks["simple"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )


def test_mnist_notebook_run(notebooks):
    notebook_path = notebooks["mnist"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )


def test_01_notebook_run(notebooks):
    notebook_path = notebooks["01_training_introduction"]

    # remove fridge_object and fridge_object.zip in data_dir since the notebook uses this dataset, and we're not overwritting it
    data_dir = os.path.join(path_notebooks(), os.pardir, "data")
    fridge_objects_data_dir = os.path.join(data_dir, "fridgeObjects")
    fridge_objects_zip = os.path.join(data_dir, "fridgeObjects.zip")
    if os.path.exists(fridge_objects_data_dir):
        shutil.rmtree(fridge_objects_data_dir)
    if os.path.exists(fridge_objects_zip):
        os.remove(fridge_objects_zip)

    # test on recycle dataset
    data_path = unzip_url(Urls.recycle_path, overwrite=True)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=data_path),
        kernel_name=KERNEL_NAME,
    )
