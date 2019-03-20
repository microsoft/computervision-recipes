#!/usr/bin/env python
# coding: utf-8

# # Testing different Hyperparameters

# Lets say we want to learn more about __how different learning rates and different image sizes affect our model's accuracy when restricted to 10 epochs__, and we want to build an experiment to test out these hyperparameters.
#
# In this notebook, we'll walk through how we use out Parameter Sweeper module with the following:
#
# - use python to perform this experiment
# - use the CLI to perform this experiment
# - evalute the results using Pandas

# ---

# In[1]:


import sys

sys.path.append("../")
import os

from utils_ic.common import ic_root_path
from utils_ic.datasets import unzip_url, Urls
from utils_ic.parameter_sweeper import *


# Lets download some data that we want to test on. To use the Parameter Sweeper tool for single label classification, we'll need to make sure that the data is stored such that images are sorted into their classes inside of a subfolder. In this notebook, we'll use the Fridge Objects dataset provided in `utils_ic.datasets.Urls`, which is stored in the correct format.

# In[2]:


input_data = unzip_url(Urls.fridge_objects_path, exist_ok=True)


# ## Using Python

# We start by creating the Parameter Sweeper object:

# In[3]:


sweeper = ParameterSweeper()


# Before we start testing, it's a good idea to see what the default parameters Are. We can use a the property `parameters` to easily see those default values.

# In[4]:


sweeper.parameters


# Now that we know the defaults, we can pass it the parameters we want to test.
#
# In this notebook, we want to see the effect of different learning rates across different image sizes using only 8 epochs (the default number of epochs is 15). To do so, I would run the `update_parameters` functions as follows:
#
# ```python
# sweeper.update_parameters(learning_rate=[1e-3, 1e-4, 1e-5], im_size=[299, 499], epochs=[10])
# ```
#
# Notice that all parameters must be passed in as a list, including single values such the number of epochs.
#
# These parameters will be used to calculate the number of permutations to run. In this case, we've pass in three options for learning rates, two for image sizes, and one for number of epochs. This will result in 3 X 2 X 1 total permutations (in otherwords, 6 permutations).

# In[5]:


sweeper.update_parameters(
    learning_rate=[1e-3, 1e-4, 1e-5], im_size=[299, 499], epochs=[10]
)


# Now that we have our parameters defined, we call the `run()` function with the dataset to test on.
#
# We can also optionally pass in:
# - the number of repetitions to run each permutation (default is 3)
# - whether or not we want the training to stop early if the metric (accuracy) doesn't improve by 0.01 (1%) over 3 epochs (default is False)
#
# The `run` function returns a multi-index dataframe which we can work with right away.

# In[6]:


df = sweeper.run(datasets=[input_data], reps=3)
df


# ## Using the CLI

# Instead of using python to run this experiment, we may want to test from the CLI. We can do so by using the `scripts/benchmark.py` file.
#
# First we move up to the `/image_classification` directory.

# In[7]:


os.chdir(ic_root_path())


# To reproduce the same test (different learning rates across different image sizes using only 8 epochs), and the same settings (3 repetitions, and no early_stopping) we can run the following:
#
# ```sh
# python scripts/sweep.py
#     --learning-rates 1e-3 1e-4 1e-5
#     --im-size 99 299
#     --epochs 5
#     --repeat 3
#     --no-early-stopping
#     --inputs <my-data-dir>
#     --output lr_bs_test.csv
# ```
#
# Additionally, we've added an output parameter, which will automatically dump our dataframe into a csv file.
#
# To simplify the command, we can use the acryonyms of the params. We can also remove `--no-early-stopping` as that is the default behavior.
#
# ```sh
# python scripts/sweep.py -lr 1e-3 1e-4 1e-5 -is 99 299 -e 5 -i <my-data-dir> -o lr_bs_test.csv
# ```

# In[8]:


# use {sys.executable} instead of just running `python` to ensure the command is executed using the environment cvbp
get_ipython().system(
    "{sys.executable} scripts/sweep.py -lr 1e-3 1e-4 1e-5 -is 99 299 -e 5 -i {input_data} -o data/lr_bs_test.csv"
)


# Once the script completes, load the csv into a dataframe to explore it's contents. We'll want to specify `index_col=[0, 1, 2]` since it is a multi-index dataframe.
#
# ```python
# df = pd.read_csv("data/lr_bs_test.csv", index_col=[0, 1, 2])
# ```

# HINT: You can learn more about how to use the script with the `--help` flag.

# In[14]:


get_ipython().system("{sys.executable} scripts/sweep.py --help")


# ---

# ## Visualizing our results

# When we read in out multi-index dataframe, index 0 represents the run number, index 1 represents a single permutation of parameters, and index 2 represents the dataset.

# To see the results, show the df using the `clean_df` helper function. This will display all the hyperparameters in a nice, readable way.

# In[15]:


df = clean_df(df)
df


# Since we've run our benchmarking over 3 repetitions, we may want to just look at the averages across the different __run numbers__.

# In[16]:


df.mean(level=(1, 2)).T


# Additionally, we may want simply to see which set of hyperparameters perform the best across the different __datasets__. We can do that by averaging the results of the different datasets. (The results of this step will look similar to the above since we're only passing in one dataset).

# In[17]:


df.mean(level=(1)).T


# To make it easier to see which permutation did the best, we can plot the results using the `plot_df` helper function. This plot will help us easily see which parameters offer the highest accuracies.

# In[18]:


plot_df(df.mean(level=(1)), sort_by="accuracy")
