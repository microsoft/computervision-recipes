# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import glob
import papermill as pm
import pytest
import scrapbook as sb
import shutil

from utils_cv.common.gpu import linux_with_gpu

# Parameters
KERNEL_NAME = "cv"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
def test_01_notebook_run(classification_notebooks):
    if linux_with_gpu():
        notebook_path = classification_notebooks["01_training_introduction"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(PM_VERSION=pm.__version__),
            kernel_name=KERNEL_NAME,
        )

        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
        assert len(nb_output.scraps["training_accuracies"].data) == 10
        assert nb_output.scraps["training_accuracies"].data[-1] > 0.80
        assert nb_output.scraps["validation_accuracy"].data > 0.80


@pytest.mark.notebooks
def test_02_notebook_run(classification_notebooks):
    if linux_with_gpu():
        notebook_path = classification_notebooks["02_multilabel_classification"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(PM_VERSION=pm.__version__),
            kernel_name=KERNEL_NAME,
        )

        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
        assert len(nb_output.scraps["training_accuracies"].data) == 10
        assert nb_output.scraps["training_accuracies"].data[-1] > 0.80
        assert nb_output.scraps["acc_hl"].data > 0.80
        assert nb_output.scraps["acc_zol"].data > 0.5


@pytest.mark.notebooks
def test_03_notebook_run(classification_notebooks):
    if linux_with_gpu():
        notebook_path = classification_notebooks["03_training_accuracy_vs_speed"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(PM_VERSION=pm.__version__),
            kernel_name=KERNEL_NAME,
        )

        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
        assert len(nb_output.scraps["training_accuracies"].data) == 12
        assert nb_output.scraps["training_accuracies"].data[-1] > 0.80
        assert nb_output.scraps["validation_accuracy"].data > 0.80


@pytest.mark.notebooks
def test_11_notebook_run(classification_notebooks, tiny_ic_data_path):
    if linux_with_gpu():
        notebook_path = classification_notebooks["11_exploring_hyperparameters"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(
                PM_VERSION=pm.__version__,

                # Speed up testing since otherwise would take ~12 minutes on V100
                DATA=[tiny_ic_data_path],
                REPS=1,
                IM_SIZES=[60,100],
            ),
            kernel_name=KERNEL_NAME,
        )

        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
        assert nb_output.scraps["nr_elements"].data == 6
        assert nb_output.scraps["max_accuray"].data > 0.5
        assert nb_output.scraps["min_accuray"].data < 0.5
        assert nb_output.scraps["max_duration"].data > 1.2 * nb_output.scraps["min_duration"].data


@pytest.mark.notebooks
def test_12_notebook_run(classification_notebooks):
    if linux_with_gpu():
        notebook_path = classification_notebooks["12_hard_negative_sampling"]
        pm.execute_notebook(
            notebook_path,
            OUTPUT_NOTEBOOK,
            parameters=dict(PM_VERSION=pm.__version__),
            kernel_name=KERNEL_NAME,
        )

        nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
        assert len(nb_output.scraps["train_acc"].data) == 12
        assert nb_output.scraps["train_acc"].data[-1] > 0.80
        assert nb_output.scraps["valid_acc"].data[-1] > 0.80
        assert len(nb_output.scraps["negative_sample_ids"].data) > 0
