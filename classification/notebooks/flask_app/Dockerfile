FROM continuumio/miniconda3:latest
# Download an existing image with conda installed and set up

WORKDIR /home/flask_app
# Create and move into the "/home/flask_app/" folder
# This is now the working directory

COPY docker_environment.yaml ./
COPY application.py ./
COPY templates/index.html ./templates/
COPY templates/template.html ./templates/
COPY boot.sh ./
# Copy all relevant files into the working directory

RUN chmod +x boot.sh
# Convert "boot.sh" into an executable file

RUN conda env create -f docker_environment.yaml
# Create the conda environment from the requirements listed in the docker_environment.yaml file

RUN echo "source activate mycondaenv" > ~/.bashrc
# Activate the conda environment and add this command to the .bashrc file

ENV PATH /opt/conda/envs/mycondaenv/bin:$PATH
# Set an environment path to the conda environment

EXPOSE 5000
# Expose port 5000, i.e. port through which to interact with the Flask app

ENTRYPOINT ["./boot.sh"]
# Execute the "./boot.sh" command once all the above has been set up
# Reference: http://www.easy-analysis.com/dockerizing-python-flask-app-and-conda-environment/
