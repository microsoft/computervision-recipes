# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import numpy as np
import os
import pytest
import torch
import urllib.request
import random
import requests

from PIL import Image
from torch import tensor
from pathlib import Path
from fastai.vision import (
    cnn_learner,
    unet_learner,
    DatasetType,
    get_image_files,
    get_transforms,
    models,
    SegmentationItemList,
)
from fastai.vision.data import ImageList, imagenet_stats
from typing import List, Tuple
from tempfile import TemporaryDirectory

from .resources import coco_sample
from utils_cv.common.data import unzip_url
from utils_cv.common.gpu import db_num_workers
from utils_cv.classification.data import Urls as ic_urls
from utils_cv.detection.data import Urls as od_urls
from utils_cv.detection.bbox import DetectionBbox, AnnotationBbox
from utils_cv.detection.dataset import DetectionDataset
from utils_cv.detection.model import (
    get_pretrained_fasterrcnn,
    get_pretrained_maskrcnn,
    get_pretrained_keypointrcnn,
    DetectionLearner,
    _extract_od_results,
    _apply_threshold,
)
from utils_cv.segmentation.data import Urls as seg_urls
from utils_cv.segmentation.dataset import load_im, load_mask
from utils_cv.segmentation.model import (
    confusion_matrix,
    get_ratio_correct_metric,
    predict,
)
from utils_cv.similarity.data import Urls as is_urls
from utils_cv.similarity.model import compute_features_learner
from utils_cv.action_recognition.data import Urls as ar_urls
from utils_cv.action_recognition.dataset import (
    VideoDataset,
    get_transforms as ar_get_transforms,
    get_default_tfms_config
)

storage_url = "https://cvbp-secondary.z19.web.core.windows.net/"


def path_classification_notebooks():
    """ Returns the path of the classification notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "scenarios",
            "classification",
        )
    )


def path_similarity_notebooks():
    """ Returns the path of the similarity notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "scenarios",
            "similarity",
        )
    )


def path_detection_notebooks():
    """ Returns the path of the detection notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, "scenarios", "detection"
        )
    )


def path_action_recognition_notebooks():
    """ Returns the path of the action recognition notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "scenarios",
            "action_recognition",
        )
    )


def path_segmentation_notebooks():
    """ Returns the path of the similarity notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "scenarios",
            "segmentation",
        )
    )


# ----- Module fixtures ----------------------------------------------------------


@pytest.fixture(scope="module")
def classification_notebooks():
    folder_notebooks = path_classification_notebooks()

    # Path for the notebooks
    paths = {
        "00": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01": os.path.join(folder_notebooks, "01_training_introduction.ipynb"),
        "02": os.path.join(
            folder_notebooks, "02_multilabel_classification.ipynb"
        ),
        "03": os.path.join(
            folder_notebooks, "03_training_accuracy_vs_speed.ipynb"
        ),
        "10": os.path.join(folder_notebooks, "10_image_annotation.ipynb"),
        "11": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters.ipynb"
        ),
        "12": os.path.join(
            folder_notebooks, "12_hard_negative_sampling.ipynb"
        ),
        "20": os.path.join(folder_notebooks, "20_azure_workspace_setup.ipynb"),
        "21": os.path.join(
            folder_notebooks,
            "21_deployment_on_azure_container_instances.ipynb",
        ),
        "22": os.path.join(
            folder_notebooks, "22_deployment_on_azure_kubernetes_service.ipynb"
        ),
        "23": os.path.join(
            folder_notebooks, "23_aci_aks_web_service_testing.ipynb"
        ),
        "24": os.path.join(
            folder_notebooks, "24_exploring_hyperparameters_on_azureml.ipynb"
        ),
    }
    return paths


@pytest.fixture(scope="module")
def similarity_notebooks():
    folder_notebooks = path_similarity_notebooks()

    # Path for the notebooks
    paths = {
        "00": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01": os.path.join(
            folder_notebooks, "01_training_and_evaluation_introduction.ipynb"
        ),
        "02": os.path.join(folder_notebooks, "02_state_of_the_art.ipynb"),
        "11": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters.ipynb"
        ),
        "12": os.path.join(folder_notebooks, "12_fast_retrieval.ipynb"),
    }
    return paths


@pytest.fixture(scope="module")
def detection_notebooks():
    folder_notebooks = path_detection_notebooks()

    # Path for the notebooks
    paths = {
        "00": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01": os.path.join(folder_notebooks, "01_training_introduction.ipynb"),
        "02": os.path.join(folder_notebooks, "02_mask_rcnn.ipynb"),
        "03": os.path.join(folder_notebooks, "03_keypoint_rcnn.ipynb"),
        "04": os.path.join(
            folder_notebooks, "04_coco_accuracy_vs_speed.ipynb"
        ),
        "11": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters_on_azureml.ipynb"
        ),
        "12": os.path.join(
            folder_notebooks, "12_hard_negative_sampling.ipynb"
        ),
        "20": os.path.join(
            folder_notebooks, "20_deployment_on_kubernetes.ipynb"
        ),
    }
    return paths


@pytest.fixture(scope="module")
def action_recognition_notebooks():
    folder_notebooks = path_action_recognition_notebooks()

    # Path for the notebooks
    paths = {
        "00": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01": os.path.join(folder_notebooks, "01_training_introduction.ipynb"),
        "02": os.path.join(folder_notebooks, "02_training_hmbd.ipynb"),
        "10": os.path.join(folder_notebooks, "10_video_transformation.ipynb"),
    }
    return paths


@pytest.fixture(scope="module")
def segmentation_notebooks():
    folder_notebooks = path_segmentation_notebooks()

    # Path for the notebooks
    paths = {
        "01": os.path.join(folder_notebooks, "01_training_introduction.ipynb"),
        "11": os.path.join(folder_notebooks, "11_exploring_hyperparameters.ipynb"),
    }
    return paths


# ----- Function fixtures ----------------------------------------------------------


@pytest.fixture(scope="function")
def tmp(tmp_path_factory):
    """Create a function-scoped temp directory.
    Will be cleaned up after each test function.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Pytest default fixture

    Returns:
        str: Temporary directory path
    """
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="function")
def func_tiny_od_data_path(tmp_session) -> str:
    """ Returns the path to the fridge object detection dataset. """
    return unzip_url(
        od_urls.fridge_objects_tiny_path,
        fpath=f"{tmp_session}/tmp",
        dest=f"{tmp_session}/tmp",
        exist_ok=True,
    )


# ----- Session fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def tmp_session(tmp_path_factory):
    """ Same as 'tmp' fixture but with session level scope. """
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


# ------|-- Classification ---------------------------------------------


@pytest.fixture(scope="session")
def tiny_ic_multidata_path(tmp_session) -> List[str]:
    """ Returns the path to multiple dataset. """
    return [
        unzip_url(
            ic_urls.fridge_objects_watermark_tiny_path,
            fpath=tmp_session,
            dest=tmp_session,
            exist_ok=True,
        ),
        unzip_url(
            ic_urls.fridge_objects_tiny_path,
            fpath=tmp_session,
            dest=tmp_session,
            exist_ok=True,
        ),
    ]


@pytest.fixture(scope="session")
def tiny_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        ic_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_multilabel_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        ic_urls.multilabel_fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def multilabel_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        ic_urls.multilabel_fridge_objects_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_ic_negatives_path(tmp_session) -> str:
    """ Returns the path to the tiny negatives dataset. """
    return unzip_url(
        ic_urls.fridge_objects_negatives_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_ic_databunch(tmp_session):
    """ Returns a databunch object for the tiny fridge objects dataset. """
    im_paths = unzip_url(
        ic_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )
    return (
        ImageList.from_folder(im_paths)
        .split_by_rand_pct(valid_pct=0.1, seed=20)
        .label_from_folder()
        .transform(size=50)
        .databunch(bs=16, num_workers=db_num_workers())
        .normalize(imagenet_stats)
    )


@pytest.fixture(scope="session")
def multilabel_result():
    """ Fake results to test evaluation metrics for multilabel classification. """
    y_pred = torch.tensor(
        [
            [0.9, 0.0, 0.0, 0.0],
            [0.9, 0.0, 0.9, 0.9],
            [0.0, 0.9, 0.0, 0.0],
            [0.9, 0.9, 0.0, 0.0],
        ]
    ).float()
    y_true = torch.tensor(
        [[1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0]]
    ).float()
    return y_pred, y_true


@pytest.fixture(scope="session")
def model_pred_scores(tiny_ic_databunch):
    """Return a simple learner and prediction scores on tiny ic data"""
    model = models.resnet18
    lr = 1e-4
    epochs = 1

    learn = cnn_learner(tiny_ic_databunch, model)
    learn.fit(epochs, lr)
    return learn, learn.get_preds()[0].tolist()


@pytest.fixture(scope="session")
def testing_im_list(tmp_session):
    """ Set of 5 images from the can/ folder of the Fridge Objects dataset
     used to test positive example rank calculations"""
    im_paths = unzip_url(
        ic_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )
    can_im_paths = os.listdir(os.path.join(im_paths, "can"))
    can_im_paths = [
        os.path.join(im_paths, "can", im_name) for im_name in can_im_paths
    ][0:5]
    return can_im_paths


@pytest.fixture(scope="session")
def testing_databunch(tmp_session):
    """ Builds a databunch from the Fridge Objects
    and returns its validation component that is used
    to test comparative_set_builder"""
    im_paths = unzip_url(
        ic_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )
    can_im_paths = os.listdir(os.path.join(im_paths, "can"))
    can_im_paths = [
        os.path.join(im_paths, "can", im_name) for im_name in can_im_paths
    ][0:5]
    random.seed(642)
    data = (
        ImageList.from_folder(im_paths)
        .split_by_rand_pct(valid_pct=0.2, seed=20)
        .label_from_folder()
        .transform(size=300)
        .databunch(bs=16, num_workers=db_num_workers())
        .normalize(imagenet_stats)
    )

    validation_bunch = data.valid_ds

    return validation_bunch


# ------|-- Detection -------------------------------------------------------------


@pytest.fixture(scope="session")
def od_cup_path(tmp_session) -> str:
    """ Returns the path to the downloaded cup image. """
    im_url = storage_url + "images/cvbp_cup.jpg"
    im_path = os.path.join(tmp_session, "example.jpg")
    urllib.request.urlretrieve(im_url, im_path)
    return im_path


@pytest.fixture(scope="session")
def od_cup_mask_path(tmp_session) -> str:
    """ Returns the path to the downloaded cup mask image. """
    im_url = storage_url + "images/cvbp_cup_mask.png"
    im_path = os.path.join(tmp_session, "example_mask.png")
    urllib.request.urlretrieve(im_url, im_path)
    return im_path


@pytest.fixture(scope="session")
def od_cup_anno_bboxes(tmp_session, od_cup_path) -> List[AnnotationBbox]:
    return [
        AnnotationBbox(
            left=61,
            top=59,
            right=273,
            bottom=244,
            label_name="cup",
            label_idx=0,
            im_path=od_cup_path,
        )
    ]


@pytest.fixture(scope="session")
def od_cup_det_bboxes(tmp_session, od_cup_path) -> List[DetectionBbox]:
    return [
        DetectionBbox(
            left=61,
            top=59,
            right=273,
            bottom=244,
            label_name="cup",
            label_idx=0,
            im_path=od_cup_path,
            score=0.99,
        )
    ]


@pytest.fixture(scope="session")
def od_mask_rects() -> Tuple:
    """ Returns synthetic mask and rectangles ([left, top, right, bottom]) for
    object detection.
    """
    height = width = 100

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:10, :20] = 1
    mask[20:40, 30:60] = 2
    # corresponding binary masks of the mask above
    binary_masks = np.zeros((2, height, width), dtype=np.bool)
    binary_masks[0, :10, :20] = True
    binary_masks[1, 20:40, 30:60] = True
    # corresponding rectangles of the mask above
    rects = [[0, 0, 19, 9], [30, 20, 59, 39]]
    # a completely black image
    im = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return binary_masks, mask, rects, im


@pytest.fixture(scope="session")
def tiny_od_data_path(tmp_session) -> str:
    """ Returns the path to the fridge object detection dataset. """
    return unzip_url(
        od_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_od_mask_data_path(tmp_session) -> str:
    """ Returns the path to the fridge object detection mask dataset. """
    return unzip_url(
        od_urls.fridge_objects_mask_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_od_keypoint_data_path(tmp_session) -> str:
    """ Returns the path to the fridge object detection keypoint dataset. """
    return unzip_url(
        od_urls.fridge_objects_keypoint_milk_bottle_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def od_sample_im_anno(tiny_od_data_path) -> Tuple[Path, ...]:
    """ Returns an annotation and image path from the tiny_od_data_path fixture.
    Specifically, using the paths for 1.xml and 1.jpg
    """
    anno_path = Path(tiny_od_data_path) / "annotations" / "1.xml"
    im_path = Path(tiny_od_data_path) / "images" / "1.jpg"
    return anno_path, im_path


@pytest.fixture(scope="session")
def od_data_path_labels() -> List[str]:
    return ["water_bottle", "can", "milk_bottle", "carton"]


@pytest.fixture(scope="session")
def od_sample_raw_preds():
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    boxes = [
        [109.0, 190.0, 205.0, 408.0],
        [340.0, 326.0, 465.0, 549.0],
        [214.0, 181.0, 315.0, 460.0],
        [215.0, 193.0, 316.0, 471.0],
        [109.0, 209.0, 209.0, 420.0],
    ]

    # construct masks
    masks = np.zeros((len(boxes), 1, 666, 499), dtype=np.float)
    for rect, mask in zip(boxes, masks):
        left, top, right, bottom = [int(x) for x in rect]
        # first line of the bounding box
        mask[:, top, left : (right + 1)] = 0.05
        # other lines of the bounding box
        mask[:, (top + 1) : (bottom + 1), left : (right + 1)] = 0.7

    # construct keypoints
    start_points = [[120, 200], [350, 350], [220, 300], [250, 400], [100, 350]]
    keypoints = []
    for x, y in start_points:
        points = []
        for i in range(13):
            points.append([x + i, y + i, 2])
        keypoints.append(points)

    return [
        {
            "boxes": tensor(boxes, device=device, dtype=torch.float),
            "labels": tensor(
                [3, 3, 3, 2, 1], device=device, dtype=torch.int64
            ),
            "scores": tensor(
                [0.9985, 0.9979, 0.9945, 0.1470, 0.0903],
                device=device,
                dtype=torch.float,
            ),
            "masks": tensor(masks, device=device, dtype=torch.float),
            "keypoints": tensor(keypoints, device=device, dtype=torch.float32),
        }
    ]


@pytest.fixture(scope="session")
def od_sample_detection(od_sample_raw_preds, od_detection_mask_dataset):
    labels = ["one", "two", "three", "four"]
    detections = _extract_od_results(
        _apply_threshold(od_sample_raw_preds[0], threshold=0.001),
        labels,
        od_detection_mask_dataset.im_paths[0],
    )
    detections["idx"] = 0
    del detections["keypoints"]
    return detections


@pytest.fixture(scope="session")
def od_sample_keypoint_detection(
    od_sample_raw_preds, tiny_od_detection_keypoint_dataset
):
    labels = ["one", "two", "three", "four"]
    detections = _extract_od_results(
        _apply_threshold(od_sample_raw_preds[0], threshold=0.9),
        labels,
        tiny_od_detection_keypoint_dataset.im_paths[0],
    )
    detections["idx"] = 0
    del detections["masks"]
    return detections


@pytest.fixture(scope="session")
def od_detection_dataset(tiny_od_data_path):
    """ returns a basic detection dataset. """
    return DetectionDataset(tiny_od_data_path)


@pytest.fixture(scope="session")
def od_detection_mask_dataset(tiny_od_mask_data_path):
    """ returns a basic detection mask dataset. """
    return DetectionDataset(
        tiny_od_mask_data_path, mask_dir="segmentation-masks"
    )


@pytest.fixture(scope="session")
def tiny_od_detection_keypoint_dataset(tiny_od_keypoint_data_path):
    """ returns a basic detection keypoint dataset. """
    return DetectionDataset(
        tiny_od_keypoint_data_path,
        keypoint_meta={
            "labels": [
                "lid_left_top",
                "lid_right_top",
                "lid_left_bottom",
                "lid_right_bottom",
                "left_bottom",
                "right_bottom",
            ],
            "skeleton": [
                [0, 1],
                [0, 2],
                [1, 3],
                [2, 3],
                [2, 4],
                [3, 5],
                [4, 5],
            ],
            "hflip_inds": [1, 0, 3, 2, 5, 4],
        },
    )


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detection_learner(od_detection_dataset):
    """ returns a basic detection learner that has been trained for one epoch. """
    model = get_pretrained_fasterrcnn(
        num_classes=len(od_detection_dataset.labels) + 1,
        min_size=100,
        max_size=200,
        rpn_pre_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=250,
        rpn_post_nms_top_n_train=500,
        rpn_post_nms_top_n_test=250,
    )
    learner = DetectionLearner(od_detection_dataset, model=model)
    learner.fit(1)
    return learner


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detection_mask_learner(od_detection_mask_dataset):
    """ returns a mask detection learner that has been trained for one epoch. """
    model = get_pretrained_maskrcnn(
        num_classes=len(od_detection_mask_dataset.labels) + 1,
        min_size=100,
        max_size=200,
        rpn_pre_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=250,
        rpn_post_nms_top_n_train=500,
        rpn_post_nms_top_n_test=250,
    )
    learner = DetectionLearner(od_detection_mask_dataset, model=model)
    learner.fit(1)
    return learner


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detection_keypoint_learner(tiny_od_detection_keypoint_dataset):
    """ returns a keypoint detection learner that has been trained for one epoch. """
    model = get_pretrained_keypointrcnn(
        num_classes=len(tiny_od_detection_keypoint_dataset.labels) + 1,
        num_keypoints=len(
            tiny_od_detection_keypoint_dataset.keypoint_meta["labels"]
        ),
        min_size=100,
        max_size=200,
        rpn_pre_nms_top_n_train=500,
        rpn_pre_nms_top_n_test=250,
        rpn_post_nms_top_n_train=500,
        rpn_post_nms_top_n_test=250,
    )
    learner = DetectionLearner(tiny_od_detection_keypoint_dataset, model=model)
    learner.fit(1, skip_evaluation=True)
    return learner


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detection_eval(od_detection_learner):
    """ returns the eval results of a detection learner after one epoch of training. """
    return od_detection_learner.evaluate()


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detection_mask_eval(od_detection_mask_learner):
    """ returns the eval results of a detection learner after one epoch of training. """
    return od_detection_mask_learner.evaluate()


@pytest.mark.gpu
@pytest.fixture(scope="session")
def od_detections(od_detection_dataset):
    """ returns output of the object detector for a given test set. """
    learner = DetectionLearner(od_detection_dataset)
    return learner.predict_dl(od_detection_dataset.test_dl, threshold=0)


# ------|-- Action Recognition ------------------------------------------------


@pytest.fixture(scope="session")
def ar_vid_path(tmp_session) -> str:
    """ Returns the path to the downloaded cup image. """
    drinking_url = ar_urls.drinking_path
    vid_path = os.path.join(tmp_session, "drinking.mp4")
    urllib.request.urlretrieve(drinking_url, vid_path)
    return vid_path


@pytest.fixture(scope="session")
def ar_milk_bottle_path(tmp_session) -> str:
    """ Returns the path of the milk bottle action dataset. """
    return unzip_url(
        ar_urls.milk_bottle_action_minified_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def ar_milk_bottle_dataset(ar_milk_bottle_path) -> VideoDataset:
    """ Returns an instance of a VideoDatset built using the milk bottle dataset. """
    conf = get_default_tfms_config(train=True)
    conf.set("input_size", 28)
    conf.set("im_scale", 32)
    train_tfms = ar_get_transforms(tfms_config=conf)
    return VideoDataset(ar_milk_bottle_path, train_transforms=train_tfms)


@pytest.fixture(scope="session")
def ar_milk_bottle_split_files(tmp_session) -> VideoDataset:
    """ Returns an instance of a VideoDatset built using the milk bottle dataset. """
    r = requests.get(ar_urls.milk_bottle_action_test_split)
    test_split_file_path = os.path.join(
        tmp_session, "milk_bottle_action_test_split.txt"
    )
    with open(test_split_file_path, "wb") as f:
        f.write(r.content)

    r = requests.get(ar_urls.milk_bottle_action_train_split)
    train_split_file_path = os.path.join(
        tmp_session, "milk_bottle_action_train_split.txt"
    )
    with open(train_split_file_path, "wb") as f:
        f.write(r.content)

    return (train_split_file_path, test_split_file_path)


@pytest.fixture(scope="session")
def ar_milk_bottle_dataset_with_split_file(
    ar_milk_bottle_path, ar_milk_bottle_split_files,
) -> VideoDataset:
    """ Returns an instance of a VideoDataset built using the milk bottle
    dataset and custom split files. """
    train_split_file_path = ar_milk_bottle_split_files[0]
    test_split_file_path = ar_milk_bottle_split_files[1]
    conf = get_default_tfms_config(train=True)
    conf.set("input_size", 28)
    conf.set("im_scale", 32)
    train_tfms = ar_get_transforms(tfms_config=conf)
    return VideoDataset(
        ar_milk_bottle_path,
        train_split_file=train_split_file_path,
        test_split_file=test_split_file_path,
        train_transforms=train_tfms
    )


# ----- AML Settings ----------------------------------------------------------


@pytest.fixture(scope="session")
def coco_sample_path(tmpdir_factory) -> str:
    """ Returns the path to a coco-formatted annotation. """
    path = tmpdir_factory.mktemp("data").join("coco_sample.json")
    path.write_text(coco_sample, encoding=None)
    return path


# TODO i can't find where this function is being used
def pytest_addoption(parser):
    parser.addoption(
        "--subscription_id",
        help="Azure Subscription Id to create resources in",
    )
    parser.addoption("--resource_group", help="Name of the resource group")
    parser.addoption("--workspace_name", help="Name of Azure ML Workspace")
    parser.addoption(
        "--workspace_region", help="Azure region to create the workspace in"
    )


@pytest.fixture
def subscription_id(request):
    return request.config.getoption("--subscription_id")


@pytest.fixture
def resource_group(request):
    return request.config.getoption("--resource_group")


@pytest.fixture
def workspace_name(request):
    return request.config.getoption("--workspace_name")


@pytest.fixture
def workspace_region(request):
    return request.config.getoption("--workspace_region")


# @pytest.fixture(scope="session")
# def testing_im_list(tmp_session):
#     """ Set of 5 images from the can/ folder of the Fridge Objects dataset
#      used to test positive example rank calculations"""
#     im_paths = unzip_url(
#         Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True
#     )
#     can_im_paths = os.listdir(os.path.join(im_paths, "can"))
#     can_im_paths = [
#         os.path.join(im_paths, "can", im_name) for im_name in can_im_paths
#     ][0:5]
#     return can_im_paths


# ------|-- Similarity ---------------------------------------------


@pytest.fixture(scope="session")
def tiny_is_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        is_urls.fridge_objects_retrieval_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )


@pytest.fixture(scope="session")
def tiny_ic_databunch_valid_features(tiny_ic_databunch):
    """ Returns DNN features for the tiny fridge objects dataset. """
    learn = cnn_learner(tiny_ic_databunch, models.resnet18)
    embedding_layer = learn.model[1][6]
    features = compute_features_learner(
        tiny_ic_databunch, DatasetType.Valid, learn, embedding_layer
    )
    return features


# ------|-- Segmentation ---------------------------------------------


@pytest.fixture(scope="session")
def tiny_seg_data_path(tmp_session, seg_classes) -> str:
    """ Returns the path to the segmentation tiny fridge objects dataset. """
    path = unzip_url(
        seg_urls.fridge_objects_tiny_path,
        fpath=tmp_session,
        dest=tmp_session,
        exist_ok=True,
    )
    classes_path = Path(path) / "classes.txt"
    with open(classes_path, "w") as f:
        for c in seg_classes:
            f.write(c + "\n")
    return path


@pytest.fixture(scope="session")
def tiny_seg_databunch(tiny_seg_data_path, seg_classes):
    """ Returns a databunch object for the segmentation tiny fridge objects dataset. """
    get_gt_filename = (
        lambda x: f"{tiny_seg_data_path}/segmentation-masks/{x.stem}.png"
    )
    return (
        SegmentationItemList.from_folder(tiny_seg_data_path)
        .split_by_rand_pct(valid_pct=0.1, seed=10)
        .label_from_func(get_gt_filename, classes=seg_classes)
        .transform(get_transforms(), tfm_y=True, size=50)
        .databunch(bs=8, num_workers=db_num_workers())
        .normalize(imagenet_stats)
    )


@pytest.fixture(scope="session")
def seg_classes() -> List[str]:
    """ Returns the segmentation class names. """
    return ["background", "can", "carton", "milk_bottle", "water_bottle"]


@pytest.fixture(scope="session")
def seg_classes_path(tiny_seg_data_path) -> str:
    """ Returns the path to file with class names. """
    return Path(tiny_seg_data_path) / "classes.txt"


@pytest.fixture(scope="session")
def seg_im_mask_paths(tiny_seg_data_path) -> str:
    """ Returns path to images and their corresponding masks. """
    im_dir = Path(tiny_seg_data_path) / "images"
    mask_dir = Path(tiny_seg_data_path) / "segmentation-masks"
    im_paths = sorted(get_image_files(im_dir))
    mask_paths = sorted(get_image_files(mask_dir))
    return im_paths, mask_paths


@pytest.fixture(scope="session")
def seg_im_and_mask(seg_im_mask_paths) -> str:
    """ Returns a single image with its mask. """
    im = load_im(seg_im_mask_paths[0][0])
    mask = load_mask(seg_im_mask_paths[1][0])
    return im, mask


@pytest.fixture(scope="session")
def seg_learner(tiny_seg_databunch, seg_classes):
    return unet_learner(
        tiny_seg_databunch,
        models.resnet18,
        wd=1e-2,
        metrics=get_ratio_correct_metric(seg_classes),
    )


@pytest.fixture(scope="session")
def seg_prediction(seg_learner, seg_im_and_mask):
    return predict(seg_im_and_mask[0], seg_learner)


@pytest.fixture(scope="session")
def seg_confusion_matrices(seg_learner, tiny_seg_databunch):
    return confusion_matrix(seg_learner, tiny_seg_databunch.valid_dl)
