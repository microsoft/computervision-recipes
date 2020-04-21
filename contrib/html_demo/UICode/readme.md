
## HTML Demo - UI Files

### Directory Description

This directory contains a html file with separate stylesheet and JavaScript functions. 


| File name | Description |
| --- | --- |
| [example_imgs.js](example_imgs.js) | Static definitions used to display DNN model Example output |
| [index.html](index.html) | User interface components |
| [style.css](style.css) | Styling of the components on the webpage |
| [script.js](script.js) | JavaScript functions to drive the back-end of the webpage |




### Usage

The files in this repository are made up of the user interface components with functioning back-end. User Interface "Use My Model" tab allows you to upload multiple image files, test images with your DNN model's API and visualize the output of the model. "See Example" tab allows you to visualize the output of three machine learning model scenarios (image classification, object detection, and image similarity) on a set of example images.


To run a webpage, please follow the guidelines on [html_demo/readme.md](../readme.md) for a necessary set up. You have to execute the notebooks in JupyterCode in your conda environment [JupyterCode/readme.md](../JupyterCode/readme.md) and deploy the models to be able to visualize different machine learning models on "See Example" tab of the webpage.


[style.css](style.css) and [script.js](script.js) have to be in the same directory as [index.html](index.html) to allow accurate full-rendering of the webpage.
