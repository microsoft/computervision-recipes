# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests


import papermill as pm

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


def test_01_notebook_run(notebooks, dataset):
    notebook_path = notebooks["01_training_introduction"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, DATA_PATH=dataset),
        kernel_name=KERNEL_NAME,
    )


def test_02_notebook_run(notebooks, dataset):
    notebook_path = notebooks["02_training_accuracy_vs_speed"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=dataset,
            MODEL_TYPE="fast_inference",  # options: ['fast_inference', 'high_accuracy', 'small_size']
            EPOCHS_HEAD=1,
            EPOCHS_BODY=1,
        ),
        kernel_name=KERNEL_NAME,
    )


def test_11_notebook_run(notebooks, dataset):
    notebook_path = notebooks["11_exploring_hyperparameters"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA=[dataset],
            REPS=1,
            LEARNING_RATES=[1e-3],
            IM_SIZES=[199],
            EPOCHS=[1],
        ),
        kernel_name=KERNEL_NAME,
    )
