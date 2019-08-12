# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

import os
import pytest
import torch
from fastai.vision import cnn_learner, models
from fastai.vision.data import ImageList, imagenet_stats
from typing import List
from tempfile import TemporaryDirectory
from utils_cv.common.data import unzip_url
from utils_cv.classification.data import Urls


def path_classification_notebooks():
    """ Returns the path of the classification notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "classification",
            "notebooks",
        )
    )


def path_similarity_notebooks():
    """ Returns the path of the similarity notebooks folder. """
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            "similarity",
            "notebooks",
        )
    )


# ----- Module fixtures ----------------------------------------------------------


@pytest.fixture(scope="module")
def classification_notebooks():
    folder_notebooks = path_classification_notebooks()

    # Path for the notebooks
    paths = {
        "00_webcam": os.path.join(folder_notebooks, "00_webcam.ipynb"),
        "01_training_introduction": os.path.join(
            folder_notebooks, "01_training_introduction.ipynb"
        ),
        "02_multilabel_classification": os.path.join(
            folder_notebooks, "02_multilabel_classification.ipynb"
        ),
        "03_training_accuracy_vs_speed": os.path.join(
            folder_notebooks, "03_training_accuracy_vs_speed.ipynb"
        ),
        "10_image_annotation": os.path.join(
            folder_notebooks, "10_image_annotation.ipynb"
        ),
        "11_exploring_hyperparameters": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters.ipynb"
        ),
        "12_hard_negative_sampling": os.path.join(
            folder_notebooks, "12_hard_negative_sampling.ipynb"
        ),
        "20_azure_workspace_setup": os.path.join(
            folder_notebooks, "20_azure_workspace_setup.ipynb"
        ),
        "21_deployment_on_azure_container_instances": os.path.join(
            folder_notebooks,
            "21_deployment_on_azure_container_instances.ipynb",
        ),
        "22_deployment_on_azure_kubernetes_service": os.path.join(
            folder_notebooks, "22_deployment_on_azure_kubernetes_service.ipynb"
        ),
        "23_aci_aks_web_service_testing": os.path.join(
            folder_notebooks, "23_aci_aks_web_service_testing.ipynb"
        ),
        "24_exploring_hyperparameters_on_azureml": os.path.join(
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
        "11": os.path.join(
            folder_notebooks, "11_exploring_hyperparameters.ipynb"
        ),
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


# ----- Session fixtures ----------------------------------------------------------


@pytest.fixture(scope="session")
def tmp_session(tmp_path_factory):
    """ Same as 'tmp' fixture but with session level scope. """
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="session")
def tiny_ic_multidata_path(tmp_session) -> List[str]:
    """ Returns the path to multiple dataset. """
    return [
        unzip_url(
            Urls.fridge_objects_watermark_tiny_path, tmp_session, exist_ok=True
        ),
        unzip_url(Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True),
    ]


@pytest.fixture(scope="session")
def tiny_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True)


@pytest.fixture(scope="session")
def tiny_multilabel_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        Urls.multilabel_fridge_objects_tiny_path, tmp_session, exist_ok=True
    )


@pytest.fixture(scope="session")
def multilabel_ic_data_path(tmp_session) -> str:
    """ Returns the path to the tiny fridge objects dataset. """
    return unzip_url(
        Urls.multilabel_fridge_objects_path, tmp_session, exist_ok=True
    )


@pytest.fixture(scope="session")
def tiny_ic_databunch(tmp_session):
    """ Returns a databunch object for the tiny fridge objects dataset. """
    im_paths = unzip_url(
        Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True
    )
    return (
        ImageList.from_folder(im_paths)
        .split_by_rand_pct(valid_pct=0.1, seed=20)
        .label_from_folder()
        .transform(size=50)
        .databunch(bs=16)
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
        Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True
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
        Urls.fridge_objects_tiny_path, tmp_session, exist_ok=True
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
        .databunch(bs=16)
        .normalize(imagenet_stats)
    )

    validation_bunch = data.valid_ds

    return validation_bunch


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
