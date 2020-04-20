
## HTML Demo - Jupyter Code

### Director Description

This directory contains a few helper notebooks that upload files and deploy models that allow the web page to work. 

| Notebook name | Description |
| --- | --- |
| [13_image_similarity_export.ipynb](13_image_similarity_export.ipynb)| Exports computed reference image features for use in visualizng results |
| [20_upload_ui.ipynb](20_upload_ui.ipynb)| Uploads web page files to Azure Blob storage |
| [30_deployment_to_azure_app_service.ipynb](30_deployment_to_azure_app_service.ipynb)| Deploys image classification model as an Azure app service |



### Usage

These notebooks can be run in the [Microsoft Computer Vision conda environment](https://github.com/microsoft/computervision-recipes/blob/master/SETUP.md).

If you want to use an image similarity model, you can run [13_image_similarity_export.ipynb](13_image_similarity_export.ipynb) to export your image features for the web page to use.

To upload the web page for sharing, notebook [20_upload_ui.ipynb](20_upload_ui.ipynb) outlines the process of uploading to Azure Blob storage.

As the web page needs the API to allow CORS, we recommend uploading models as an Azure app service. Notebook [30_deployment_to_azure_app_service.ipynb](30_deployment_to_azure_app_service.ipynb) gives a tutorial on how to do so with an image classification model.
