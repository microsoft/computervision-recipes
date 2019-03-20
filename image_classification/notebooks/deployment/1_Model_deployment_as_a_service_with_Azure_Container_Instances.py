# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Deployment of a model as a service with Azure Container Instances
#
# At the end of this notebook, we will have learned how to:
# - Deploy this model as a web service using an [Azure Container Instance](https://azure.microsoft.com/en-us/services/container-instances/)
# - Test that the service works well when called from the notebook
#
#
# <img src="screenshots/ACI_diagram.png" width="500" align=left>

# ### Pre-requisite
#
# For this notebook to run properly on your machine, the following should already be in place:

# #### Local machine setup
# - Miniconda virtual environment - Please follow [these instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python), if you do not have one yet
# - Python 3.6 (Tensorboard and some Azure elements are not supported with Python 3.7)
# - Azure SDK: [Steps](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python#install-the-sdk-1) to follow to install it
# - Pytorch: `conda install –c pytorch pytorch torchvision`
# - Fastai: `conda install fastai=1.0.44 -c fastai –c conda-forge`
#   - Please follow [these instructions](https://github.com/fastai/fastai/blob/master/README.md#installation), if you encounter any issues with the above

# #### Azure subscription setup
# You also need an account on the Azure platform. If you do not have one:
# - [Create an account](https://azure.microsoft.com/en-us/free/services/machine-learning/)
# - [Add a subscription](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/SubscriptionsBlade) -- _You can start with a free one_
# - [Add a resource group](https://ms.portal.azure.com/#blade/HubsExtension/Resources/resourceType/Microsoft.Resources%2Fsubscriptions%2FresourceGroups)
#
# What is required here is:
# - A subscription ID
# - A resource group

# ### Library import
# Throughout this notebook, we will be using a variety of libraries. We are listing them here for better readibility.

# +
# For automatic reloading of modified libraries
# %reload_ext autoreload
# %autoreload 2

# Regular python libraries
from io import BytesIO
import os
import requests
import sys
from sys import platform

# fast.ai
from fastai.vision import *
import torchvision.models as models

# Azure
import azureml.core
from azureml.core import Experiment, Workspace
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice

# Computer Vision repository
sys.path.append("../../")
from utils_ic.datasets import Urls, unzip_url, data_path
from utils_ic.windows_create_cnn import custom_create_cnn
from utils_ic.image_conversion import image2json
# -

# ### Azure workspace

# #### SDK version
#
# Before we start, let's check which version of the Azure SDK we are working with.

# Check core SDK version number
print("Azure ML SDK Version: {}".format(azureml.core.VERSION))

# #### Workspace creation
# Now that we have our environment and proper libraries in place, let's load an existing workspace or create a new one on our Azure account, and save it to a local configuration file (`./aml_config/config.json`).
#
# If it is the first time we create a workspace, or if we are missing our `config.json` file, we need to provide the appropriate:
# - subscription ID
# - resource group
# - location
# - name of the workspace we want to create (e.g. 'myworkspace').

# +
# If we do not already have a ./aml_config/config.json file, we need to uncomment and run the 4 lines below
# subscription_id = "<our_subscription_id>"
# resource_group = "<our_resource_group>"
# workspace_name = "<our_workspace_name>" (e.g. "myworkspace")
# workspace_region = "<our_workspace_region>" (e.g. "westus2")

try:
    # Load the workspace
    if os.path.exists('./aml_config/config.json'):
        # From a configuration file
        ws = Workspace.from_config()
    else:
        # Or directly from Azure
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        # and generate a local configuration file
        ws.write_config()
    print("Workspace configuration succeeded.")
except:
    # Create a workspace from scratch
    ws = Workspace.create(name=workspace_name,
                          subscription_id=subscription_id,
                          resource_group=resource_group,
                          create_resource_group=True,
                          location=workspace_region
                         )
    print("Workspace {} is not accessible. Change your parameters or create a new workspace.")
# -

# Let's check that the workspace is properly loaded

# Print the workspace attributes
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep = '\n')

# We can see this workspace on the Azure portal by sequentially clicking on:
# - Resource groups, and clicking the one we referenced above

# <img src="screenshots/resource_group.jpg" width="800">

# - workspace_name

# <img src="screenshots/workspace.jpg" width="800">

# #### Experiment

# An experiment contains a series of trials called `Runs`. In the present case, the `run` will consist of the training of our model. Through this `run`'s methods, we will be able to log several metrics such as training and test loss and accuracy, and to tag our run. The full description of the `run` class is available [here](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py).
#
# Let's first create a new experiment. If an experiment with the same name already exists in our workspace, the run we will generate will be recorded under that already existing experiment.

# Create a new/Retrieve an existing experiment
experiment_name = 'image-classifier-webservice'
experiment = Experiment(workspace=ws, name=experiment_name)

print("New/Existing experiment:\n --> Name: {}\n --> Workspace name: {}".format(experiment.name, experiment.workspace.name))

# While our `experiment` object is now created, it is not visible on the Azure portal yet. It will become visible once the first run under it gets triggered.

# #### Run initialization
#
# With our new experiment, we can now start a run. `start_logging()` allows us to start an interactive session in our current experiment.
#
# *__Note:__ For those of us who have run several experiments and triggered many runs, we may start seeing a warning message related to a too big snapshot. This is not a problem (WAITING FOR INFO FROM SDK TEAM HERE). If that is the case, we can use the `snapshot_directory=None` argument to ignore this warning.*

# Initialize the run
runner = experiment.start_logging(snapshot_directory=None)

print("Run {} was created".format(runner.id))

# Now that we have launched our run, we can see our experiment on the Azure portal, under `Experiments` (in the left-hand side list).
#
# <img src="screenshots/experiment.jpg" width="800">

# From here, we will be able to get a visual representation of metrics tracked during training. For that, let's click on:
# - *image-classifier-webservice*, under `Experiments`
# - the run number at the top of the displayed list
#
# <img src="screenshots/experiment_runs_list.jpg" width="800">
#
# We are all set from the Azure point of view. Let's now focus on the training of our model, on our local machine.

# ### Data retrieval and pre-processing
#
# As our main purpose is to deploy our model as a service, we are going to train our model on the small fridge objects dataset, and keep our training section short. The deployment process would remain the same if we trained on more data, with a different model and for a longer time. Details on input data structure, on model misclassifications investigation, and on visualization are available in the [**01_training_introduction.ipynb**](https://github.com/Microsoft/ComputerVisionBestPractices/blob/staging/image_classification/notebooks/01_training_introduction.ipynb) notebook.
#
# Let's start by setting some parameters and downloading our dataset from an Azure blob storage.

DATA_PATH      = unzip_url(Urls.fridge_objects_path, exist_ok=True)  # Path to downloaded data
EPOCHS         = 5
LEARNING_RATE  = 1e-4
IMAGE_SIZE     = 299  # size to convert our input data into
BATCH_SIZE     = 16
ARCHITECTURE   = models.resnet50
# Any model among alexnet, resnet18/34/50/101/152, squeezenet1.0/1.1, densenet121/161/169/201, vgg16/19 should work
RANDOM_SEED    = 456
VALIDATION_PCT = 0.2  # For train-validation split: 80%/20%
MODEL_NAME     = 'im_classif_fridge_obj'  # Name we will give our model both locally and on Azure
PICKLED_MODEL_NAME = MODEL_NAME + '.pkl'

# The data are now stored locally in a *data/fridge_objects/* directory.

# Path to our data
print(DATA_PATH)
im_path = Path(DATA_PATH)

# We can now ingest them and start training our model.

data = (ImageItemList
        .from_folder(im_path) # Data location: in path and its subfolders
        .random_split_by_pct(valid_pct=VALIDATION_PCT, seed=RANDOM_SEED) # Train/valid split: with percentage of all data
        .label_from_folder() # Folder names contain the labels
        .transform(size=IMAGE_SIZE) # Size reduction of the images
        .databunch(bs=BATCH_SIZE) # Conversion to ImageDataBunch: with custom batch size
        .normalize(imagenet_stats))
        # Normalization: (x - mean)/std with mean and std dev of the ImageNet dataset:
        # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

print("Training dataset: {} samples\nTesting dataset: {} samples\nRead-in classes: {}"
      .format(len(data.train_ds.x), len(data.valid_ds.x), data.classes))

data.show_batch(rows=2, figsize=(6,6))

# ### Model training (locally)

# We consider here a ResNet50 pre-trained model, with all layers frozen, except the last one. For this, we use fast.ai's [create_cnn](https://docs.fast.ai/vision.learner.html#create_cnn) function.
#
# Note: If you are working on a Windows machine, depending on your environment setup, you may observe that `create_cnn` hangs and never completes. If that is the case, use the `custom_create_cnn()` function instead. It is part of the `utils.py` file that accompanies this notebook. To use it, just run:
#
# `import utils
# learn = custom_create_cnn(data, mdl_object, metrics=error_rate)`

if platform == 'win32':
    learn = custom_create_cnn(data, ARCHITECTURE, metrics=error_rate)
else:
    learn = create_cnn(data, ARCHITECTURE, metrics=error_rate)

# We train our model over 5 epochs and over one [learning rate cycle](https://medium.com/@nachiket.tanksale/finding-good-learning-rate-and-the-one-cycle-policy-7159fe1db5d6). **This may take a few minutes, if trained on CPU.**

learn.fit_one_cycle(EPOCHS)

# As we saw in the [01_training_introduction.py](https://github.com/Microsoft/ComputerVisionBestPractices/blob/staging/image_classification/notebooks/01_training_introduction.ipynb) notebook, we can use several fast.ai built-in functions to visualize training and validation metrcis, to identify misclassified images, to build a confusion matrix and to retrieve the worst predictions made by the model.
#
# But since we are within an experiment run, we can also capture some of these metrics on the Azure portal, by "attaching" them to our run. `runner.log()` and `runner.log_list()` help us do just that.

# +
training_losses = [x.numpy().ravel()[0] for x in learn.recorder.losses]
error_rate = [x[0].numpy().ravel()[0] for x in learn.recorder.metrics]

runner.log_list('training_loss', training_losses)
runner.log_list('validation_loss', learn.recorder.val_losses)
runner.log_list('error_rate', error_rate)
runner.log_list('learning_rate', learn.recorder.lrs)
# -

# These metrics are now part of the experiment. We can then visualize them on the Azure portal, by clicking on the latest run number or by running the `runner.get_portal_run()` command below.
#
# <img src="screenshots/experiment_run_recorded_metrics.jpg" width="800">
#
# Note that the "learning_rate" plot indeed shows the variation of this metric over one cycle only.

print(runner.get_portal_url())

# ### Model export
#
# To be able to use the model we just trained, we need to export it to our local machine. We store it in an `outputs/` subfolder.

current_directory = os.getcwd()
print(current_directory)

# +
output_folder = current_directory+'/outputs/'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

learn.export(output_folder + PICKLED_MODEL_NAME)
# -

# ### Run termination
# Now, that the model is trained, we can close the run using `runner.complete()`. Results can then be captured and finalized. `runner` returns a URL that allows us to see them on the Azure portal.

runner.complete()

runner

# ### Model deployment on Azure

# #### 1. Model registration
#
# Our final goal is to deploy the model we trained as a web service. To do so, we need to first register it in our workspace, i.e. place it in our workspace's model registry. We do this by using `runner.upload_file()` and `runner.register_model()`, which takes:
# - a `model_name` that represents well what our model does
# - and the `model_path` of the model relative to the run.
#
# Other methods exist to load the model and register it all at once, but using `runner.upload_file()` and specifying the `outputs/` folder allow us to check the presence of the uploaded model on the Azure portal.

runner.upload_file(name='outputs/' + PICKLED_MODEL_NAME, path_or_stream=current_directory+'/outputs/' + PICKLED_MODEL_NAME)

# Register the model with the workspace
model = runner.register_model(model_name=MODEL_NAME, model_path='outputs/' + PICKLED_MODEL_NAME)
# !!! We need to make sure that the model name we use here is the same as in the scoring script below !!!

# Now that the model is uploaded and registered, we can see it on the Azure platform, under `Outputs` and `Models`
#
# <img src="screenshots/uploaded_model.jpg" width="800">
# <img src="screenshots/models.jpg" width="800">

# We can also check that it is programatically accessible

print("Model:\n --> Name: {}\n --> ID: {}\n --> Path:{}"
      .format(model.name, model.id, model._get_model_path_remote(model.name, model.version, ws)))

runner.get_file_names()

# If we are also interested in verifying which model we uploaded, we can download it to our local machine

model.download()

# #### 2. Scoring script
# For the web service to return predictions on a given input image, we need to provide it with instructions on how to use the model we just registered. These instructions are stored in the scoring script.
#
# This script must contain two required functions, `init()` and `run(input_data)`:
# - In the `init()` function, we typically load the model into a global object. This function is executed only once when the Docker container is started.
# - In the `run(input_data)` function, the model is used to predict a value based on the input data. The input and output to `run` typically use JSON as serialization and de-serialization format but we are not limited to that.
#
# This file must also be stored in the current directory.

scoring_script = "score.py"

# + {"magic_args": "$scoring_script", "language": "writefile"}
# # Copyright (c) Microsoft. All rights reserved.
# # Licensed under the MIT license.
#
# from base64 import b64encode, b64decode
# from io import BytesIO
# import json
# import numpy as np
# from azureml.core.model import Model
# from fastai.vision import *
# from fastai.vision import Image as FImage
#
# def init():
#     global model
#     model_path = Model.get_model_path(model_name='im_classif_fridge_obj')
#     # ! We cannot use MODEL_NAME here otherwise the execution on Azure will fail !
#     
#     actual_path, actual_file = os.path.split(model_path)
#     model = load_learner(path=actual_path, fname=actual_file)
#
#
# def run(raw_data):
#
#     # Expects raw_data to be a list within a json file
#     result = []
#     all_data = [b64decode(im) for im in json.loads(raw_data)['data']]
#     
#     for im_bytes in all_data:
#         try:
#             im = open_image(BytesIO(im_bytes))
#             pred_class, pred_idx, outputs = model.predict(im)
#             result_dict = {"label": str(pred_class), "probability": str(float(outputs[pred_idx]))}
#             result.append(result_dict)
#         except Exception as e:
#             result_dict = {"label": str(e), "probability": ''}
#             result.append(result_dict)
#     return result
#
# -

# #### 3. Environment setup

# In order to make predictions on the Azure platform, it is critical to create an environment similar to the one we used to train the model. For this, we use a [Docker container](https://www.docker.com/resources/what-container). We configure it via a yaml file that will contain all the conda dependencies we relied on on our local machine.

# +
# Create an empty conda environment and add the scikit-learn package
conda_filename = "myenv.yml"  # this file can only be save in the current directory

conda_env = CondaDependencies()
conda_env.add_conda_package("scikit-learn")
conda_env.add_channel("pytorch")
conda_env.add_channel("conda-forge")
conda_env.add_channel("fastai")
conda_env.add_conda_package("pytorch")
conda_env.add_conda_package("torchvision")
conda_env.add_conda_package("spacy=2.0.18")
conda_env.add_conda_package("fastai=1.0.44")
conda_env.add_conda_package("dataclasses")

# Display the environment
print(conda_env.serialize_to_string())

# Write the environment to disk
conda_env.save_to_file(base_directory='./', conda_file_path=conda_filename)
# -

# There are different ways of creating a Docker image on Azure. Here, we create it separately from the service it will be used for. This way of proceeding gives us direct access to the Docker image object. Thus, if the service deployment fails, but the Docker image gets deployed successfully, we can try deploying the service again, without having to create a new image all over again.

# Docker image configuration
image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "Image with fast.ai Resnet50 model (fastai 1.0.44)",
                                                  tags = {'area': "tableware", 'type': "CNN resnet50"}
                                                 )

# Docker image creation
docker_image = ContainerImage.create(name = "image-classif-resnet50-f44", # this is the model object
                                     models = [model],
                                     image_config = image_config,
                                     workspace = ws)
# The image name should not contain more than 32 characters, and should not contain any spaces, dots or underscores

# %%time
docker_image.wait_for_creation(show_output = True)  # This can take up to 15 min

print(ws.images["image-classif-resnet50-f44"].image_build_log_uri)

# #### 4. Computational resources

# In this notebook, we use [Azure Container Instances](https://docs.microsoft.com/en-us/azure/container-instances/) which are good for quick and cost-effective development/test deployment scenarios.
#
# To set them up properly, we need to indicate the number of cores and the amount of memory we want to allocate to our web service. Optional tags and descriptions are also available for us to identify the instances in AzureML when viewing the `Compute` tab in the Azure Portal.
#
# **Note:** For production workloads, it is better to use [Azure Kubernetes Service (AKS)](https://docs.microsoft.com/en-us/azure/aks/) instead. We will demonstrate how to so this in the next notebook (to be published).

# Create a deployment configuration with 1 CPU and 5 gigabyte of RAM
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                memory_gb=5, 
                                                tags={'webservice': 'image classification model (fastai1.0.44)'}, 
                                                description='This service classifies fridge object images into 4 different groups.')

# #### 5. Web service deployment

# The final step to deploying our web service is to call `WebService.deploy_from_image()`. This function uses the Docker image and the deployment configuration we created above to perform the following:
#
# - Deploy the docker image to an Azure Container Instance
# - Call the `init()` function in our scoring file
# - Provide an HTTP endpoint for scoring calls
#
# The `deploy_from_image` method requires the following parameters:
#
# - workspace: the workspace containing the service
# - name: a unique named used to identify the service in the workspace
# - image: a docker image object that contains the environment needed for scoring/inference
# - deployment_config: a configuration object describing the compute type
#
# Azure Container Instances have no associated ComputeTarget, so we do not specify any here. Remember, we already provided information on the number of CPUs and the amount of memory needed in the service configuration file above.
#
# > Note: The web service creation can take **several minutes**.

# Creation of the Docker container image and deployment of the web service
service_name = 'im-classif-websvc'
service = Webservice.deploy_from_image(workspace=ws,
                                       name=service_name,
                                       image=docker_image,
                                       deployment_config=aci_config)

# An alternative way of deploying the service is to deploy from the model directly. In that case, we would need to provide the docker_image configuration object, not the image object itself.
# The advantage of `deploy_from_image` over `deploy_from_model` is that the former allows us
# to re-use the same Docker image in case the deployment of this service fails, or even for other
# types of deployments, as we will see in the next notebook (to be pushlished).

# +
# # # Web service creation from registered model
# # Image configuration object indicating how our deployment container needs to be created
# image_config = ContainerImage.image_configuration(execution_script=scoring_script,
#                                                   runtime="python", 
#                                                   conda_file="myenv.yml")
# service = Webservice.deploy_from_model(name=service_name,
#                                        workspace=ws,
#                                        image_config=image_config,
#                                        deployment_config=aci_config,
#                                        models=[model])
# -

service.wait_for_deployment(show_output=True)

# When successful, we expect to see the following:
#
# `Creating image
# Image creation operation finished for image <service_name>, operation "Succeeded"
# Creating service
# Running .....
# SucceededACI service creation operation finished, operation "Succeeded"`
#
# In the case where the deployment is not successful, we can look at the image and service logs to debug. [These instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-troubleshoot-deployment) can also be helpful.

# +
# Access to the service logs
# print(service.get_logs())
# -

print("Service {} is _{}_ and available at {}".format(service.name, service.state, service.scoring_uri))

# We can also check the presence and status of both our new Docker image and web service on the Azure portal, under the `Images` and `Deployments` tabs, respectively.
#
#
# <img src="screenshots/docker_images.jpg" width="800">
# <img src="screenshots/deployments.jpg" width="800">

# #### 6. Testing of the service

# To make sure that our service is working as expected, let's test it. We first need to retrieve a test image and to pre-process it into the format expected by our model.

# ##### 6a. Using the `run` API

# A service typically expects input data to be in a JSON serializable format, which fast.ai's image objects are not. Here, we use our own `image2json()` function to transform these images into lists of floats and then into strings.

images_fname_list = [os.path.join('test_images', 'im_11.jpg'), os.path.join('test_images', 'im_97.jpg')]
test_samples = image2json(images_fname_list, current_directory)

# Prediction using the deployed model
result = service.run(test_samples)

# Plots of the results
actual_labels = ['milk_bottle', 'water_bottle']
for k in range(0, len(result)):
    title = "{}/{} - {}%".format(actual_labels[k], result[k]['label'], 
                                 round(100.*float(result[k]['probability']), 2))
    open_image(images_fname_list[k]).show(title=title)

# ##### 6b. Via a raw HTTP request

# +
# Send the same test data
headers = {'Content-Type':'application/json'}

resp = requests.post(service.scoring_uri, test_samples, headers=headers)

print("POST to url", service.scoring_uri)
print("prediction:", resp.text)
# -

# ##### 6c. Using a user interface

# As we discussed above, Azure Container Instances are typically used to develop and test deployments. They are however not suited for production, or calls to the service outside of the run context. For this to be possible, we need to deploy to Azure Kubernetes Service. An example of this will be available in the next notebook (to be published), where we will show how to create a web app with a user interface.

# ### Clean up

# #### Service termination
#
# In a real-life scenario, it is likely that the service we created would need to be up and running at all times. However, in the present demonstrative case, and now that we have verified that our service works, we can delete it.

service.delete()
