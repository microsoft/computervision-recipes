# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests


import papermill as pm
from utils_ic.datasets import Urls, unzip_url

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
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=unzip_url(Urls.recycle_path, exist_ok=True),
        ),
        kernel_name=KERNEL_NAME,
    )
