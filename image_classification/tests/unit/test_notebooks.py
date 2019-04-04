# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests


# import glob
# import os
import papermill as pm

# import shutil
from utils_ic.datasets import Urls, unzip_url

# Unless manually modified, python3 should be
# the name of the current jupyter kernel
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


# def test_deploy_1_notebook_run(notebooks):
#     notebook_path = notebooks["deploy_on_ACI"]
#     pm.execute_notebook(
#         notebook_path,
#         OUTPUT_NOTEBOOK,
#         parameters=dict(
#             PM_VERSION=pm.__version__,
#             DATA_PATH=unzip_url(Urls.fridge_objects_path, exist_ok=True),
#         ),
#         kernel_name=KERNEL_NAME,
#     )
#     try:
#         os.remove("myenv.yml")
#     except OSError:
#         pass
#     try:
#         os.remove("score.py")
#     except OSError:
#         pass
#
#     try:
#         os.remove("output.ipynb")
#     except OSError:
#         pass
#
#     # There should be only one file, but the name may be changed
#     file_list = glob.glob("./*.pkl")
#     for filePath in file_list:
#         try:
#             os.remove(filePath)
#         except OSError:
#             pass
#
#     shutil.rmtree(os.path.join(os.getcwd(), "azureml-models"))
#     shutil.rmtree(os.path.join(os.getcwd(), "models"))
#     shutil.rmtree(os.path.join(os.getcwd(), "outputs"))
