
# Crowd counting

This repository provides production ready version of crowd counting algorithms. The different algorithms are unified under a set of consistent APIs. 

## At a glance
[![Figure 1][pic 1]][pic 1]

Note: All sample images for the crowd counting scenario are from www.unsplash.com.

While there's a wide range of crowd counting models, two practical matters need to be accounted for:
- Speed. To support near real time reporting, the model should run fast enough. 
- Crowd density. We need to allow for both high-density and low-density scenarios for the same camera. Most crowd counting models were trained using high density datasets and they tend not to work well for low density scenarios. On the other hand, models like Faster-RCNN work well for low density crowd but not so much for high density scenarios. 

Based on evaluation of multiple implementations of Crowd Counting models on our propietary dataset, we narrowed down the models to two options: the Multi Column CNN model (MCNN) from [this repo](https://github.com/svishwa/crowdcount-mcnn) and the OpenPose model from [this repo](https://github.com/jiajunhua/ildoonet-tf-pose-estimation). Both models met our speed requirements. 
- For high density crowd images, the MCNN model delivered good results. 
- For low density scenarios, OpenPose performed well. 
- When crowd density if unknown beforehand, we use a heuristic approach: the prediction from MCNN is used if the following conditions are met: OpenPose prediction is above 20 and MCNN is above 50. Otherwise, the OpenPose prediction used. The thresholds for the models can be changed depending on your scenario.

[pic 1]: media/obs_vs_pred.PNG

## Setup
### Dependencies
You need dependencies below. 
- Python 3
- Tensorflow 1.4.1+
- PyTorch

### Install
Clone the repo recursively and install libraries.
```bash
git clone --recursive git@github.com:microsoft/ComputerVision.git
cd ComputerVision/contrib/crowd_counting/
pip install -r requirements.txt 
```

Then download the MCNN model trained on the Shanghai Tech A dataset and save it under folder crowdcounting/data/models/ of the cloned repo. The link to the model can be found in the Test section of [this repo](https://github.com/svishwa/crowdcount-mcnn).

### Test
Below is how to run the demo app and call the service using a local image.
```
python crowdcounting/demo/app-start.py -p crowdcounting/data/models/mcnn_shtechA_660.h5
curl -H "Content-type: application/octet-stream" -X POST http://0.0.0.0:5000/score --data-binary @/path/to/image.jpg
```
## Performance
Below we report mean absolute error on our own dataset. 

|Crowd Density | MCNN | OpenPose | Router|
| -------| ------- | ------- | ------- |
| low | 51.95 | 0.56 | 0.63 |
| high |  151.11 | 442.15 | 195.93 |


## Examples
A tutorial can be found in the crowdcounting/examples folder.

## Docker image
A docker image for a demo can be built and run with the following commands:
```bash
nvidia-docker build -t crowd-counting:mcnn-openpose-gpu
nvidia-docker run -d -p 5000:5000 crowd-counting:mcnn-openpose-gpu
```
Then type the url 0.0.0.0:5000 in a browser to try the demo.

## Build Status
[![Build Status](https://dev.azure.com/team-sharat/crowd-counting/_apis/build/status/lixzhang.cnt?branchName=lixzhang%2Fsubmodule-rev3)](https://dev.azure.com/team-sharat/crowd-counting/_build/latest?definitionId=49&branchName=lixzhang%2Fsubmodule-rev3)
