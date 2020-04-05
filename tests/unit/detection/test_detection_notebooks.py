# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This test is based on the test suite implemented for Recommenders project
# https://github.com/Microsoft/Recommenders/tree/master/tests

import papermill as pm
import pytest
import scrapbook as sb

from utils_cv.common.data import unzip_url
from utils_cv.detection.data import Urls

# Unless manually modified, python3 should be
# the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.mark.notebooks
def test_00_notebook_run(detection_notebooks):
    notebook_path = detection_notebooks["00"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["detection_bounding_box"].data) > 0


@pytest.mark.gpu
@pytest.mark.notebooks
def test_01_notebook_run(detection_notebooks, tiny_od_data_path):
    notebook_path = detection_notebooks["01"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_data_path,
            EPOCHS=1,
            IM_SIZE=100,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_losses"].data) > 0
    training_aps = nb_output.scraps["training_average_precision"].data
    assert len(training_aps) > 0
    for d in training_aps:
        assert isinstance(d, dict)
    assert len(set([len(d) for d in training_aps])) == 1


@pytest.mark.gpu
@pytest.mark.notebooks
def test_02_notebook_run(detection_notebooks, tiny_od_mask_data_path):
    notebook_path = detection_notebooks["02"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_mask_data_path,
            EPOCHS=1,
            IM_SIZE=100,
        ),
        kernel_name=KERNEL_NAME,
    )
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["training_losses"].data) > 0
    training_aps = nb_output.scraps["training_average_precision"].data
    assert len(training_aps) > 0
    for d in training_aps:
        assert isinstance(d, dict)
    assert len(set([len(d) for d in training_aps])) == 1


@pytest.mark.gpu
@pytest.mark.notebooks
def test_03_notebook_run(
    detection_notebooks, tiny_od_keypoint_data_path, tmp_session
):
    notebook_path = detection_notebooks["03"]
    data_path2 = unzip_url(
        Urls.fridge_objects_keypoint_top_bottom_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            IM_SIZE=100,
            EPOCHS=1,
            DATA_PATH=tiny_od_keypoint_data_path,
            DATA_PATH2=data_path2,
            THRESHOLD=0.01,
        ),
        kernel_name=KERNEL_NAME,
    )
    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["keypoints"].data) == len(
        nb_output.scraps["bboxes"].data
    )


@pytest.mark.gpu
@pytest.mark.notebooks
def test_04_notebook_run(detection_notebooks, tiny_od_data_path):
    notebook_path = detection_notebooks["04"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_data_path,
            LABELS=["can", "carton", "milk_bottle", "water_bottle"]*21 #coco model was pre-trained on 80 classes 
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["aps"].data) == 2
    assert nb_output.scraps["num_test_images"].data == 38


@pytest.mark.gpu
@pytest.mark.notebooks
def test_12_notebook_run(
    detection_notebooks, tiny_od_data_path, tiny_ic_negatives_path
):
    notebook_path = detection_notebooks["12"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(
            PM_VERSION=pm.__version__,
            DATA_PATH=tiny_od_data_path,
            NEG_DATA_PATH=tiny_ic_negatives_path,
            EPOCHS=1,
            IM_SIZE=100,
        ),
        kernel_name=KERNEL_NAME,
    )

    nb_output = sb.read_notebook(OUTPUT_NOTEBOOK)
    assert len(nb_output.scraps["valid_accs"].data) == 1
    assert 5 <= len(nb_output.scraps["hard_im_scores"].data) <= 10
