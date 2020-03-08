# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import scrapbook as sb

# Parameters
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_01_notebook_run(detection_notebooks):
    epochs = 3
    notebook_path = detection_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    skip_evaluation = nb_output.scraps["skip_evaluation"].data
    training_losses = nb_output.scraps["training_losses"].data
    assert len(training_losses) == epochs
    assert training_losses[-1] < 0.5
    if not skip_evaluation:
        training_aps = nb_output.scraps["training_average_precision"].data
        assert len(training_aps) == epochs
        for d in training_aps[-1].values():
            assert d > 0.5


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_02_notebook_run(detection_notebooks):
    epochs = 5
    notebook_path = detection_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    training_losses = nb_output.scraps["training_losses"].data
    assert len(training_losses) == epochs
    assert training_losses[-1] < 0.85
    training_aps = nb_output.scraps["training_average_precision"].data
    assert len(training_aps) == epochs
    for d in training_aps[-1].values():
        assert d > 0.15


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_03_notebook_run(detection_notebooks):
    epochs = 5
    notebook_path = detection_notebooks["03"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=epochs),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    training_losses = nb_output.scraps["training_losses"].data
    assert len(training_losses) == epochs
    assert training_losses[0] > 1.5 * training_losses[-1]
    assert len(nb_output.scraps["keypoints"].data) == len(
        nb_output.scraps["bboxes"].data
    )


@pytest.mark.notebooks
@pytest.mark.linuxgpu
def test_12_notebook_run(detection_notebooks):
    notebook_path = detection_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__, EPOCHS=3),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert nb_output.scraps["valid_accs"].data[-1] > 0.5
    assert len(nb_output.scraps["valid_accs"].data) == 1
    assert len(nb_output.scraps["hard_im_scores"].data) == 10
