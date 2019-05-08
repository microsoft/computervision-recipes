#!/usr/bin/env python
# coding: utf-8

# <i>Copyright (c) Microsoft Corporation. All rights reserved.</i>
#
# <i>Licensed under the MIT License.</i>
#
#
# # Interacting with a web service through a user interface
#
# ## Table of content
# 1. [Introduction](#intro)
# 1. [Flask application](#flask-app)
#   1. [Using the user interface directly](#local)
#     1. [From the terminal window](#terminal)
# 	1. [From this notebook](#ntbk)
#   1. [Using the user interface through a Docker container](#docker)
#     1. [Dockerization of our application](#container)
#     1. [Testing](#test)
# 1. [Service monitoring](#insights)
# 1. [Clean up](#clean)
#   1. [Docker resources deletion](#del_docker)
#   1. [Application Insights deactivation and web service termination](#del_app_insights)
#   1. [Web service Docker image deletion](#del_image)
# 1. [Next steps](#next-steps)
# 1. [Resources](#resources)

# ## 1. Introduction <a id="intro"/>
# As part of our deployment series (i.e. notebooks in "2x_"), we are focusing here on the user experience. In this tutorial, we will indeed learn how to create a user interface using Flask, which:
# - Calls our AKS web service
# - Allows our users to easily upload several images
# - Returns a table of the uploaded images along with their respective predicted classes and associated probabilities.
#
# We are assuming that we already have a web service that serves our machine learning model on AKS. If that is not the case, we can refer to the [AKS deployment notebook](https://github.com/microsoft/ComputerVision/blob/master/classification/notebooks/22_deployment_on_azure_kubernetes_service.ipynb) to create one.
#
# In this tutorial, we will also use Docker to containerize our Flask application. If we don't have it installed on our machine, we can follow the instructions below to do so:
#   - [Linux](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
#   - [MacOS](https://docs.docker.com/docker-for-mac/install/)
#   - [Windows](https://docs.docker.com/docker-for-windows/install/).
#
# We will then be able to run our Flask application directly, and within a Docker container, on our local machine.

# ## 2. Flask application  <a id="flask-app">
#
# This notebook is accompanied by 3 files, stored in the `flask_app/` folder:
# - application.py: python code that orchestrates the functioning of the user interface -- More particularly, it:
#         - controls the display of the user interface (cf. index.html)
#         - pre-processes the images uploaded by the user
#         - sends these images to the web service
#         - collects its responses
#         - renders them in a webpage (cf. template.html)
# - templates/index.html: html document which controls the look and feel of the user interface
# - templates/template.html: html document which pipes the images uploaded by the user and the response from the web service (i.e. classes and probabilities) into a table and displays them in a webpage.
#
# These 3 files construct a Flask application that will allow us to interact with our web service.
#
# <i><b>Note:</b> In this section, we will use the web service hosted on AKS. This will be helpful when we deploy our Flask application on the same AKS cluster (cf. next notebook).</i>

# ### 2.A Using the user interface directly <a id="local"/>
#
# We can run this application in 2 different ways:
# 1. From a terminal window, in our conda environment
# 2. From within this notebook
#
# #### 2.A.a From the terminal window <a id="terminal"/>
# To run the Flask application from our local machine, we need to:
# - Change directory to `flask_app/`
# - Open the `application.py` file and replace the `<service_uri>` and `<primary_key>` values by the values we obtained in section 3.B of the [web service testing](https://github.com/microsoft/ComputerVision/blob/staging/classification/notebooks/23_aci_aks_web_service_testing.ipynb) notebook. These values can also be found on the Azure portal:
#     - In our workspace, let's click on "Deployments"
#     - There, let's click on our web service: "aks-cpu-image-classif-web-svc"
#     - And copy and paste the content of "Scoring URI" and "Primary key" into our application.py -- The former should be of the form `https://xxx.xxx.xxx.xxx/api/v1/service/aks-cpu-image-classif-web-svc/score`, and the latter should be a set of 32 letters and numbers
# - Run `python application.py`
#
# This returns a URL (typically `http://127.0.0.1:5000`). Clicking on it brings us to a file uploader webpage. Let's upload some images we have on our local machine or that we downloaded from the internet.
#
# If our service works as expected, we should see the results presented in a table.
#
# <img src="media/file_uploader_webpage.jpg" width="500" align="left">
# <img src="media/predictions.jpg" width="400" align="center"/>
#
# <i><b>Notes:</b>
# - Depending on the size of the uploaded images, the service may or may not provide a response. It is best to send images of a few kB each.
# - The uploader function creates an uploads/ folder in our working directory, which contains the images we uploaded.</i>
#
# Let's now turn the application off to run it from within the notebook. To do this, we just need to hit Ctrl+C in our terminal.
#
# #### 2.A.b From this notebook <a id="ntbk"/>
# Here, we use a built-in magic command `%run`. The experience is then the same.

# In[1]:


# Change directory
get_ipython().run_line_magic("cd", "flask_app")


# In[2]:


# Built-in magic command to run our Flask application
get_ipython().run_line_magic("run", "-i application.py")


# Let's click on the url and upload images several times, so we can better see the server's response in the next section.
#
# To end the test, we just need to click on the "Stop" (square) button in the notebook menu.

# ### 2.B Using the user interface through a Docker container <a id="docker">
#
# While having a local application is nice, it is sometimes much better to have a website that can be accessed from any machine, and by more than one person. We may even want to "productionize" our application. In that case, we need to:
# - Containerize this application
# - Register the container in a container registry
# - Deploy this container on AKS nodes.
#
# In the rest of this notebook, we will focus on the containerization part. We will proceed with the registration and deployment of the Docker image on Azure in the next notebook.
#
# For now, we will use Docker, and leverage the additional files that are in the `flask_app/` folder:
# 1. `Dockerfile` contains the instructions to follow to create our Docker image:
#     - Download an existing Docker image with conda installed and set up
#     - Create and move into the "/home/flask_app/" folder
#     - Copy all relevant files into the working directory
#     - Create the conda environment from the requirements listed in the docker_environment.yaml file
#     - Activate the conda environment and add this command to the .bashrc file
#     - Expose port 5000, i.e. port through which to interact with the Flask app
#     - Execute the "./boot.sh" command (cf. below), once all the above has been set up
# 2. `docker_environment.yaml` contains all the libraries to set up in our conda environment for our application to run properly
# 3. `aks_deployment.yaml` specifies how to create a new deployment for our Flask app to run on (e.g. Docker image used, port, number of replicas, IP address, etc.)
# 4. `boot.sh` is a script that starts the [Gunicorn web server](https://gunicorn.org/) and identifies the `application.py` file as the one containing the Flask app we want to run.
#
#
# <i><b>Note:</b> In this section, we will execute command line operations. To run them easily in this notebook, we added a "!" in front of them all. If we want to run these commands in a regular terminal, we just need to remove that "!".</i>
#
# #### 2.B.a Dockerization of our application <a id="container">
#
# First, let's check that we have Docker running

# In[ ]:


get_ipython().system("docker ps")


# If it is, we should at least see a line of headers such as:
#
# `
# CONTAINER ID       IMAGE       COMMAND       CREATED       STATUS       PORTS       NAMES
# `
#
# If we do not see this result, we need to install Docker, as decribed in the [1. Introduction](#intro) section above.
#
# Now that Docker is running, let's create a Docker container, a Docker image and deploy our application into it. In this tutorial, we will call our Docker image `flaskappdockerimage`.
#
# To create a container and an image inside it, we use the [docker build](https://docs.docker.com/v17.09/engine/reference/commandline/build/) command, which refers to our `Dockerfile`. This generates a Docker image, which contains the files relevant to the Flask app, as well as a conda environment to run the app in.
#
# <i><b>Note:</b> This takes <b>~5 minutes</b> as the downloading of the needed python libraries and the creation of the conda environment are rather slow.</i>

# In[ ]:


get_ipython().system("docker build -t flaskappdockerimage:latest .")


# We are now ready to start our Docker image. The default port for Flask is 5000, so this is what we are using here. Note that we also exposed that port in our `Dockerfile`.

# In[ ]:


get_ipython().system("docker run -d -p 5000:5000 flaskappdockerimage")


# Running `docker container list -a` shows us the list of all Docker containers and images we have running (or not) on our machine. An important column to look at, in the returned table, is "STATUS". If the creation of the Docker image completed successfully, we should see "Up X seconds".

# In[ ]:


get_ipython().system("docker container list -a")


# #### 2.B.b Testing <a id="test">
#
# If everything worked well, we should be able to see our application in action at `http://localhost:5000/`. As before, we should see our application front page, which invites the user to select pictures to classify. Our application is now running on our machine, but in a Docker container. This is that container that we will register on Azure, and deploy on AKS in the next notebook.
#
# ##### Debugging
# In some cases, however, this step may have failed, and the "STATUS" column may show `Exited (x)`, where `x` is an integer. Alternatively, while the image is shown as being running, the application may not behave as expected, and `http://localhost:5000` may point to nothing. When one of these situations occurs, we can look into the docker logs and investigate what is going wrong.
#
# <i><b>Note:</b> For this to be possible, we need to add the parameter `debug=True` to our application.py file:  `app.run(debug=True)`. This parameter should be removed when testing the application locally through the notebook, as we did in section 2.A.b.</i>

# In[ ]:


# When our application doesn't behave as expected
# !docker logs <container_ID>
# Replace <container_ID> by the number identifier displayed above


# For Windows users, `docker logs <container ID>` may return `standard_init_linux.go:207: exec user process caused "no such file or directory"`. This may mean that at least one of our files has bad line endings. If that is the case, we can open Git Bash and run `dos2unix.exe <filename>`. This [converts](https://www.liquidweb.com/kb/dos2unix-removing-hidden-windows-characters-from-files/) Windows "CRLF" (Carriage Return (\r), Line Feed (\n)) endings into Unix standard "LF".
#
# Once we have fixed the endings of our files, we need to delete the container and image we just created, and re-create them with our Unix understandable files.

# In[ ]:


# !docker stop <container_ID>  # stops our running container
# !docker container rm <contain_ID>  # deletes our container
# !docker image rm flaskappdockerimage  # deletes our Docker image
# !docker build -t flaskappdockerimage:latest .  # creates the Docker image
# !docker run -d -p 5000:5000 flaskappdockerimage  # starts the image


# Once our application is running in our Docker container, we can upload a few images. As before, this should return a table with the images we uploaded, and their predicted classes and probabilities.
#
# <i><b>Note:</b> The first time the service runs, it needs to be "primed", so it may take a little more time than for subsequent requests.</i>
#
# We can now call our image classifier model, through our Flask application, directly or through our Docker container.

# ## 3. Service monitoring <a id="insights"/>
#
# The web service we are using here in the one we started monitoring in section 4. of the [web service testing](https://github.com/microsoft/ComputerVision/blob/master/classification/notebooks/23_aci_aks_web_service.ipynb) notebook. As we played with our Flask app, both directly and through our Docker container, we called that service. We should consequently see some activity on the Application Insights section of our workspace, in the Azure portal.

# ## 4. Clean up <a id="clean">
#
# In the next notebook, we will register and deploy the Docker container we just created, on Azure. So, we need to keep it running as well as our web service. This explains why the commands below are commented out. However, if we were to not pursue this objective, we would delete them both, as well as all the resources we used.
#
# As in the prior notebook, we incurred a cost of about $15 a day. To get a better sense of pricing, we can refer to [this calculator](https://azure.microsoft.com/en-us/pricing/calculator/?service=virtual-machines). We can also navigate to the [Cost Management + Billing pane](https://ms.portal.azure.com/#blade/Microsoft_Azure_Billing/ModernBillingMenuBlade/BillingAccounts) on the portal, click on our subscription ID, and click on the Cost Analysis tab to check our credit usage.
#
# ### 4.A Docker resources deletion <a id="del_docker">
#
# We start with the Docker container and image that contain the Flask application.

# In[ ]:


# !docker stop <container_ID>  # stops our running container
# !docker container rm <contain_ID>  # deletes our container
# !docker image rm flaskappdockerimage  # deletes our Docker image


# ### 4.B Application Insights deactivation and web service termination <a id="del_app_insights">
#
# We can then delete resources associated with our image classifier web service.

# In[ ]:


# Telemetry deactivation
# aks_service.update(enable_app_insights=False)

# Service termination
# aks_service.delete()

# Compute target deletion
# aks_target.delete()


# ### 4.C Web service Docker image deletion <a id="del_image">
#
# We finish with the deletion of the Docker image that we created in [21_deployment_on_azure_container_instances.ipynb](https://github.com/Microsoft/ComputerVisionBestPractices/blob/staging/image_classification/notebooks/21_deployment_on_azure_container_instances.ipynb), and which contains our image classifier model.

# In[ ]:


print("Docker images:")
for docker_im in ws.images:
    print(
        f"    --> Name: {ws.images[docker_im].name}\n    --> ID: {ws.images[docker_im].id}\n    --> Tags: {ws.images[docker_im].tags}\n    --> Creation time: {ws.images[docker_im].created_time}"
    )


# In[ ]:


docker_image = ws.images["image-classif-resnet18-f48"]
# docker_image.delete()


# ## 5. Next steps <a id="next-steps">
#
# In the next notebook, we will register and deploy the Docker image we created here, on the AKS cluster that already hosts our web service.
#
#
# ## 6. Resources <a id="resources">
#
# Throughout this notebook, we used a few of tools and concepts. If that is of interest, here resources we recommend reading:
# - Docker: [General documentation](https://docs.docker.com/get-started/), [CLI functions](https://docs.docker.com/engine/reference/commandline/docker/)
# - [Dockerization of a Flask app in a conda environment (by Easy Analysis)](http://www.easy-analysis.com/dockerizing-python-flask-app-and-conda-environment/)
