# Semantic Segmentation using PyTorch and Azure Machine Learning

This subproject contains a production ready training pipeline for a semantic segmentation model using PyTorch and Azure Machine Learning.

## Installation

To install the Azure ML CLI v2, [follow these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli)

To install the last set of known working python dependencies run

```bash
pip install requirements.txt
```

Note that this project utilizes [pip-tools](https://github.com/jazzband/pip-tools) to manage its dependencies. Direct dependencies that the project requires are specified in `requirements.in` and may be upgraded to greater versions than that of `requirements.txt` at your own risk.
