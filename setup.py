# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools import setup
from os import path


UTILS_CV = "utils_cv"  # Utility folder name
# UTILS_CV_PATH = path.join(path.abspath(path.dirname(__file__)), UTILS_CV)
# README = path.join(UTILS_CV_PATH, "README.md")
README = path.join(UTILS_CV, "README.md")
exec(open(path.join(UTILS_CV, "__init__.py")).read())

with open(README, encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="utils_cv",
    version=__version__,
    description="Computer Vision Utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/ComputerVision",
    project_urls={
        "Bug Tracker": "https://github.com/microsoft/ComputerVision/issues",
        "Source Code": "https://github.com/microsoft/ComputerVision",
        "Documentation": "https://github.com/microsoft/ComputerVision",
    },
    author="CVDev Team at Microsoft",
    classifiers=[
        "Development Status :: 2-Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
    # Not fully test the dependency coverage, please make sure to sync
    # with environment.yml
    install_requires=[
        "azureml-sdk[notebooks,contrib]>=1.0.30",  # requires ipykernel, papermill, jupyter-core, jupyter-client
        "bqplot",
        "fastai==1.0.48",  # requires pytorch, torchvision, nvidia-ml-py3
        "scikit-learn>=0.19.1",
    ],
    keywords=", ".join(
        [
            "computer vision",
            "deep learning",
            "convolutional neural network",
            "image classification",
            "image similarity",
            "data science",
            "artificial intelligence",
            "machine learning",
            "gpu",
        ]
    ),
    packages=[UTILS_CV],
    python_requires="==3.6.8",
)
