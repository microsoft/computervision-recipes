FROM tensorflow/tensorflow:1.12.0-rc2-gpu-py3
LABEL maintainer="lixun.zhang@microsoft.com"
COPY . /crowd-counting
WORKDIR /crowd-counting
RUN apt-get update && \
    apt-get install -y --no-install-recommends dialog apt-utils && \
    apt-get install -y \
    build-essential \
    curl \
    libopenmpi-dev \
    libgtk2.0-dev \    
    libsm6 \
    libxext6 \
    libxrender-dev \
    python3.5 \
    python3-pip \
    python3-tk \    
    python-dev \
    git 
RUN apt-get purge python-numpy
RUN pip3 install --upgrade pip setuptools
RUN pip3 uninstall numpy -y
RUN pip3 install numpy
RUN pip3 install cython
RUN pip3 install torch==1.1.0 torchvision==0.3.0
RUN pip3 install -r requirements.txt
RUN pip3 install -e .
ENTRYPOINT ["python3"]
CMD ["./crowdcounting/demo/app-advanced.py", "-p", "crowdcounting/data/models/mcnn_shtechA_660.h5"]
