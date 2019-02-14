(This document is up-to-date as of 1/29/2019)

# Overview of Azure's Computer Vision Offerings
Microsoft provides a variety of options when it comes to computer vision. The outline below provides an overview of such services, starting with the highest level service where you simply consume an API to the lowest level service where you develop the model and the infrastructure required to deploy it.

## Cognitive Services API
Cognitive Services API allow you to consume machine learning hosted services. Within Cognitive Services API, there are several computer vision services:

- [Face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/) (for face detection and recognition)
- [Content Moderator](https://azure.microsoft.com/en-us/services/cognitive-services/content-moderator/) (for image, text and video moderation)
- [Computer Vision](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/) (for analyzing images, reading text and handwriting, identifying celebrities, and intelligently generating thumbnails)
- [Video Indexer](https://azure.microsoft.com/en-us/services/media-services/video-indexer/) (for analyzing videos)

Targeting popular and specific use cases, these services can be consumed with easy to use APIs. Users do no have to do any modeling or understand any machine learning concepts. They simply need to pass an image or video to the hosted endpoint, and consume the results that are returned.

For these Cognitive Services, the models are used are pretrained and cannot be modified. 

## Custom Vision Service
Custom Vision Service is a SaaS service where you can train your own vision models with minimal machine learning knowledge. Upload labelled training images through the browser application or through their APIs and the Custom Vision Service will help you train and evaluate your model. Once you are satisfied with your model's performance, the model will be ready for consumption as an endpoint.

Currently, the Custom Vision Service can do image classification (multi-class + multi-label) and object detection scenarios.

Learn more about the Custom Vision Service [here](https://azure.microsoft.com/en-us/services/cognitive-services/custom-vision-service/).

## Azure Machine Learning
Azure Machine Learning (AML) is a scenario-agnostic machine learning service that will help users accelerate training and deploying machine learning models. Use automated machine learning to identify suitable algorithms and tune hyperparameters faster. Improve productivity and reduce costs with autoscaling compute and DevOps for machine learning. Seamlessly deploy to the cloud and the edge with one click. Access all these capabilities from your favorite Python environment using the latest open-source frameworks, such as PyTorch, TensorFlow, and scikit-learn.

Learn more about Azure Machine Learning [here](https://azure.microsoft.com/en-us/services/machine-learning-service/).

---

# What Should I Use?
When it comes to doing computer vision on Azure, there are many options and it can be confusing to figure out what services to use.

One approach is see if the scenario you are solving for is one that is covered by one of the Cognitive Services APIs. If so, you can start by using those APIs and determine if the results are performant enough. If they are not, you may consider customizing the model with the Custom Vision Service, or building your own model using Azure Machine Learning. 

Another approach is to determine the degree of customizability and fine tuning you want. Cognitive Services APIs provide no flexibility. The Custom Vision Service provides flexibility insofar as being able to choose what kind of training data to use (it is also only limited so solving classification and object detection problems). Azure Machine Learning provides complete flexibility, letting you set hyperparameters, select model architectures (or build your own), and perform any manipulation needed at the framework (pytorch, tensorflow, cntk, etc) level. 

One consideration is that more customizability also translates to more responsibility. When using Azure Machine Learning, you get the most flexibility, but you will be responsible for making sure the models are performant and deploying them on Azure. 


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
