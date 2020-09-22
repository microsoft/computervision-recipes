
## HTML Demo - Jupyter Code

### Directory Description

This directory contains a few helper notebooks that upload files and deploy models that allow the web page to work. 

| Notebook name | Description |
| --- | --- |
| [1_image_similarity_export.ipynb](1_image_similarity_export.ipynb)| Exports computed reference image features for use in visualizng results (see details in "Image Similarity" section below) |
| [2_upload_ui.ipynb](2_upload_ui.ipynb)| Uploads web page files to Azure Blob storage |
| [3_deployment_to_azure_app_service.ipynb](3_deployment_to_azure_app_service.ipynb)| Deploys image classification model as an Azure app service |
| [4_train_and_deploy_custom_image_similarity_webapp.ipynb](4_train_and_deploy_custom_image_similarity_webapp.ipynb) | Fine-tunes a ResNet18 CNN model and deploys a custom image similarity webapp using AzureML

### Requirements

To run the code in the [2_upload_ui.ipynb](2_upload_ui.ipynb) notebook, you must first: 
1. Install the [https://pypi.org/project/azure-storage-blob/](Azure Storage Blobs client library for Python)
2. Have (or create) an Azure account with a Blob storage container where you would like to store the UI files
3. Note your Blob stoarge credentials to upload files programmatically; you will need: 
	a. Azure Account Name
	b. Azure Account Key
	c. Blob Container Name
4. Update [2_upload_ui.ipynb](2_upload_ui.ipynb) to add your Blob storage credentials

### Usage

* These notebooks can be run in the [Microsoft Computer Vision conda environment](https://github.com/microsoft/computervision-recipes/blob/master/SETUP.md).
* If you want to use an image similarity model, you can run [1_image_similarity_export.ipynb](1_image_similarity_export.ipynb) to export your image features for the web page to use.
* To upload the web page for sharing, notebook [2_upload_ui.ipynb](2_upload_ui.ipynb) outlines the process of uploading to your Azure Blob storage.
* As the web page needs the API to allow CORS, we recommend uploading models as an Azure app service. Notebook [3_deployment_to_azure_app_service.ipynb](3_deployment_to_azure_app_service.ipynb) gives a tutorial on how to do so with an image classification model.
* [4_train_and_deploy_custom_image_similarity_webapp.ipynb](4_train_and_deploy_custom_image_similarity_webapp.ipynb) guides through the process of deploying a custom image similarity web application - from finetuning a RESNET50 model using a sample dataset in ImageNet directory structure format to updating required files for the web application and deploying them along with the model.

### Image Similarity

Image similarity relies on comparing DNN features of a query image, to the respective DNN features of potentially tens of thousands of references images. The notebooks in this directory compute these reference image DNN features and package them for use in the HTML UI. The DNN features are exported into a text file and compressed to be uploaded with the HTML UI files. To compare a query image to these exported reference image features, you will need a DNN model deployed to Azure App services that is able to compute the features of the query image and return them via API call.      

Steps:
1. Execute [1_image_similarity_export.ipynb](1_image_similarity_export.ipynb) notebook to generate your reference image features and export them to compressed ZIP files
2. Execute [2_upload_ui.ipynb](2_upload_ui.ipynb) notebook to upload the HTML UI and all supporting files to your Azure Blob storage account
3. Execute [3_deployment_to_azure_app_service.ipynb](3_deployment_to_azure_app_service.ipynb) notebook to upload your model for generating DNN features for your query image and create an API endpoint in Azure App service
4. Open the index.html file from your Blob storage account in a browser, enter your API endpoint URL, upload a query image and see what you get back
5. *(Optional)* Execute [4_train_and_deploy_custom_image_similarity_webapp.ipynb](4_train_and_deploy_custom_image_similarity_webapp.ipynb) notebook to finetune a RESNET50 model and deploy a custom image similarity web application 
