#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>

# # Testing different Hyperparameters and Benchmarking

# In this notebook, we'll cover how to test different hyperparameters for a particular dataset and how to benchmark different parameters across a group of datasets using AzureML

# Similar to 11_exploring_hyperparameters.ipynb, we will learn more about __how different learning rates and different image sizes affect our model's accuracy when restricted to 10 epochs__, and we want to build an AzureML experiment to test out these hyperparameters.
#
# We present an overall process of utilizing AzureML, specifically [Hyperdrive](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive?view=azure-ml-py) component, for the hyperparameter tuning by demonstrating key steps:
# * Configure AzureML Workspace
# * Create Remote Compute Target (GPU cluster)
# * Prepare Data
# * Prepare Training Script
# * Setup and Run Hyperdrive Experiment
# * Model Import, Re-train and Test

# In[ ]:

import os
import sys

sys.path.append("../../")

from utils_cv.classification.data import Urls
from utils_cv.common.data import unzip_url

import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import azureml.data
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BanditPolicy,
    HyperDriveConfig,
    PrimaryMetricGoal,
    choice,
)
from azureml.train.estimator import Estimator

import azureml.widgets as widgets

print("SDK version:", azureml.core.VERSION)


# Ensure edits to libraries are loaded and plotting is shown in the notebook.

# In[ ]:


get_ipython().run_line_magic("reload_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")
get_ipython().run_line_magic("matplotlib", "inline")


# ### 1. Config AzureML workspace
# Below we setup AzureML workspace and get all its details as follows:

# In[ ]:


ws = Workspace.setup()
ws_details = ws.get_details()
print(
    "Name:\t\t{}\nLocation:\t{}".format(
        ws_details["name"], ws_details["location"]
    )
)


# ### 2. Create Remote Target
# We create a GPU cluster as our remote compute target. If a cluster with the same name is already exist in our workspace, the script will load it instead. We can see [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets) to learn more about setting up a compute target on different locations.
#
# This notebook selects STANDARD_NC6 virtual machine (VM) and sets it's priority as lowpriority to save the cost.

# In[ ]:


# choose a name for our cluster
cluster_name = "gpu-cluster-nc6"
# Remote compute (cluster) configuration. If you want to save the cost more, set these to small.
VM_SIZE = "STANDARD_NC6"
VM_PRIORITY = "lowpriority"

# Cluster nodes
MIN_NODES = 0
MAX_NODES = 4

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size=VM_SIZE, min_nodes=MIN_NODES, max_nodes=MAX_NODES
    )

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# we can use get_status() to get a detailed status for the current cluster.
print(compute_target.get_status().serialize())


# ### 3. Prepare data
# In this notebook, we'll use the Fridge Objects dataset, which is already stored in the correct format. We then upload our data to the AzureML workspace.
#

# In[ ]:


# Note, all the files under DATA_DIR will be uploaded to the data store
DATA = unzip_url(Urls.fridge_objects_path, exist_ok=True)
REPS = 3

ds = ws.get_default_datastore()

ds.upload(
    src_dir=os.path.dirname(DATA),
    target_path="data",
    overwrite=True,
    show_progress=True,
)


# ### 4. Prepare training script
#
# Next step is to prepare scripts that AzureML Hyperdrive will use to train and evaluate models with selected hyperparameters. To run the model notebook from the Hyperdrive Run, all we need is to prepare an entry script which parses the hyperparameter arguments, passes them to the notebook, and records the results of the notebook to AzureML Run logs.

# In[ ]:


# creating a folder for the training script here
script_folder = os.path.join(os.getcwd(), "hyperparameter")
os.makedirs(script_folder, exist_ok=True)


# In[ ]:


get_ipython().run_cell_magic(
    "writefile",
    "$script_folder/train.py",
    "\nimport numpy as np\nimport os\nfrom sklearn.externals import joblib\nimport sys\n\nimport fastai\nfrom fastai.vision import *\nfrom fastai.vision.data import *\n\nfrom azureml.core import Run\n\nrun = Run.get_context()\n\n# The datastore is mounted in the Estimator script_params which resolves to an environment variable\n# name of the format \"$AZUREML_DATAREFERENCE_XXXX\" where XXXX is the default datastore 'workspaceblobstore' in this case.\n# We retrieve our FridgeObjects dataset by giving the path such as below.\ndata_store_path = str(os.environ['AZUREML_DATAREFERENCE_workspaceblobstore'])\npath = data_store_path + '/data/fridgeObjects'\n\n# Define parameters that we are going to use for training\nIM_SIZES = [299, 499]\nARCHITECTURE  = models.resnet50\nLEARNING_RATES = [1e-3, 1e-4, 1e-5]\nEPOCHS = [10]\n\n# Getting training and validation data and training the CNN as done in 01_training_introduction.ipynb\ndata = (ImageList.from_folder(path)\n        .split_by_rand_pct(valid_pct=0.2, seed=10)\n        .label_from_folder() \n        .transform(size=299) \n        .databunch(bs=16) \n        .normalize(imagenet_stats))\n\nlearn = cnn_learner(\n    data,\n    ARCHITECTURE,\n    metrics=[accuracy]\n)\n\nlearn.unfreeze()\nlearn.fit(EPOCHS[0], LEARNING_RATES[0])\n\ntraining_losses = [x.numpy().ravel()[0] for x in learn.recorder.losses]\naccuracy = [x[0].numpy().ravel()[0] for x in learn.recorder.metrics][-1]\n\n#run.log_list('training_loss', training_losses)\n#run.log_list('validation_loss', learn.recorder.val_losses)\n#run.log_list('error_rate', error_rate)\n#run.log_list('learning_rate', learn.recorder.lrs)\nrun.log('accuracy', float(accuracy))  # Logging our primary metric 'accuracy'\n\ncurrent_directory = os.getcwd()\noutput_folder = os.path.join(current_directory, 'outputs')\nMODEL_NAME = 'im_classif_resnet50'  # Name we will give our model both locally and on Azure\nPICKLED_MODEL_NAME = MODEL_NAME + '.pkl'\nos.makedirs(output_folder, exist_ok=True)\n\nlearn.export(os.path.join(output_folder, PICKLED_MODEL_NAME))",
)


# ### 5. Setup and run Hyperdrive experiment
#
# Next step is to prepare scripts that AzureML Hyperdrive will use to train and evaluate models with selected hyperparameters. To run the model notebook from the Hyperdrive Run, all we need is to prepare an entry script which parses the hyperparameter arguments, passes them to the notebook, and records the results of the notebook to AzureML Run logs.
#
# #### 5.1 Create Experiment
# Experiment is the main entry point into experimenting with AzureML. To create new Experiment or get the existing one, we pass our experimentation name 'hyperparameter-tuning'.
#

# In[ ]:


experiment_name = "hyperparameter-tuning"
exp = Experiment(workspace=ws, name=experiment_name)


# #### 5.2. Define search space
#
# Now we define the search space of hyperparameters. For example, if you want to test different batch sizes of {64, 128, 256}, you can use azureml.train.hyperdrive.choice(64, 128, 256). To search from a continuous space, use uniform(start, end). For more options, see [Hyperdrive parameter expressions](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.parameter_expressions?view=azure-ml-py).
#
# In this notebook, we fix model architecture as models.resnet50 and the number of epochs to 10.
# In the search space, we set different learning rates and image sizes. Details about the hyperparameters can be found from our 11_exploring_hyperparameters.ipynb notebook.
#
# Hyperdrive provides three different parameter sampling methods: RandomParameterSampling, GridParameterSampling, and BayesianParameterSampling. Details about each method can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-tune-hyperparameters). Here, we use the Random Parameter sampling.

# In[ ]:


IM_SIZES = [299, 499]
LEARNING_RATES = [1e-3, 1e-4, 1e-5]

# Hyperparameter search space
param_sampling = RandomParameterSampling(
    {"learning_rate": choice(LEARNING_RATES), "im_sizes": choice(IM_SIZES)}
)

primary_metric_name = "accuracy"
primary_metric_goal = PrimaryMetricGoal.MAXIMIZE
max_total_runs = 50
max_concurrent_runs = 4

early_termination_policy = BanditPolicy(
    slack_factor=0.15, evaluation_interval=1, delay_evaluation=20
)


# <b>AzureML Estimator</b> is the building block for training. An Estimator encapsulates the training code and parameters, the compute resources and runtime environment for a particular training scenario.
# We create one for our experimentation with the dependencies our model requires as follows:
#
# ```python
# pip_packages=['fastai']
# conda_packages=['scikit-learn']
# ```

# In[ ]:


script_params = {"--data-folder": ds.as_mount()}

est = Estimator(
    source_directory=script_folder,
    script_params=script_params,
    compute_target=compute_target,
    entry_script="train.py",
    pip_packages=["fastai"],
    conda_packages=["scikit-learn"],
)
# model_run = exp.submit(est)
# widgets.RunDetails(model_run).show()


# To the Hyperdrive Run Config, we set our primary metric name and the goal (our hyperparameter search criteria), hyperparameter sampling method, and number of total child-runs. The bigger the search space, the more number of runs we will need for better results.

# In[ ]:


hyperdrive_run_config = HyperDriveConfig(
    estimator=est,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name=primary_metric_name,
    primary_metric_goal=primary_metric_goal,
    max_total_runs=max_total_runs,
    max_concurrent_runs=max_concurrent_runs,
)


# #### 5.3 Run Experiment
# Now we submit the Run to our experiment. We can see the experiment progress from this notebook by using
# ```python
# azureml.widgets.RunDetails(hyperdrive_run).show()
# ```
# or check from the Azure portal with the url link we get by running
# ```python
# hyperdrive_run.get_portal_url().```
#
# To load an existing Hyperdrive Run instead of start new one, we can use
# ```python
# hyperdrive_run = azureml.train.hyperdrive.HyperDriveRun(exp, <your-run-id>, hyperdrive_run_config=hyperdrive_run_config)
# ```
# We also can cancel the Run with
# ```python
# hyperdrive_run_config.cancel().
# ```

# In[ ]:


hyperdrive_run = exp.submit(config=hyperdrive_run_config)
widgets.RunDetails(hyperdrive_run).show()


# Once all the child-runs are finished, we can get the best run and the metrics.

# In[ ]:


# Get best run and print out metrics
best_run = hyperdrive_run.get_best_run_by_primary_metric()
best_run_metrics = best_run.get_metrics()
parameter_values = best_run.get_details()["runDefinition"]["arguments"]
best_parameters = dict(zip(parameter_values[::2], parameter_values[1::2]))

print("* Best Run Id:", best_run.id)
print(best_run)
print("\n* Best hyperparameters:")
print(best_parameters)
print("Accuracy =", best_run_metrics["accuracy"])
# print("Learning Rate =", best_run_metrics['learning_rate'])


# ### 6. Test the model
#
# We can download the best run model from outputs folder and use it to test against unseen images

# In[ ]:


current_directory = os.getcwd()
output_folder = os.path.join(current_directory, "outputs")
MODEL_NAME = "im_classif_resnet50"
PICKLED_MODEL_NAME = MODEL_NAME + ".pkl"
os.makedirs(output_folder, exist_ok=True)

for f in best_run.get_file_names():
    if f.startswith("outputs/im_classif_resnet50"):
        print("Downloading {}..".format(f))
        best_run.download_file(name=f, output_file_path=output_folder)
saved_model = load_learner(path=output_folder, file=PICKLED_MODEL_NAME)
print(saved_model)


# We can now use the retrieved best run model to get predictions on unseen images as done in 02_training_accuracy_vs_speed.ipynb notebook using
# ```python
# saved_model.predict(image)
# ```
