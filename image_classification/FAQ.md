# Image classification

## Frequently asked questions


* [TEST ](#test)
  * [How does the technology work?](#how-does-the-technology-work)
  * [Which problems can be solved using image classification, and which ones cannot?](#which-problems-can-be-solved-using-image-classification?)



### How does the technology work?

State-of-the-art image classification methods such as used in this repository are based on Convolutional Neural Networks (CNN). CNNs are a special group of Deep Learning approaches shown to work well on image data. The key is to use CNNs which were already trained on millions of images (the ImageNet dataset) and to fine-tune these pre-trained CNNs using a potentially much smaller custom dataset. This is the approach also taken in this repository. The web is full of introductions to these conceptions, such as [link](https://towardsdatascience.com/simple-introduction-to-convolutional-neural-networks-cdf8d3077bac).


### Which problems can be solved using image classification?

Image classification can be used if the object-of-interest is relatively large in the image, e.g. more than 20% image width/height. If the object is smaller, or if the location of the object is required, then object detection methods should be used instead.


<details>
<summary> How many images are required to train a model? </summary>
This depends heavily on the complexity of the problem. For example, if the object-of-interest looks very different from image to image (viewing angle, lighting condition, etc) then more training images are required for the model to learn the appearance of the object.

In practice, we have seen good results using 100 images for each class or sometime less. The only way to find out how many images are required, is by training the model using increasing number of images, while observing how the accuracy improves (while keeping the test set fixed). Once accuracy improvements become small, this would indicate that more training images are not required.
</details>

<details>
<summary> How to annotate images? </summary>
Consistency is key. For example, occluded objects should either be always annotated, or never. Furthermore, ambiguous images should be removed, eg if it is unclear to a human eye if an image shows a lemon or a tennis ball. Ensuring consistency is difficult especially if multiple people are involved, and hence our recommendation is that only a single person, the one who trains the AI model, annotates all images. This has the added benefit of gaining a better understanding of the images and of the complexity of the classification task.

Note that the test set should be of high annotation quality, so that accuracy estimates are reliable.
</details>

<details>
<summary> How to split into training and test images? </summary>
Often a random split, as is performed in the notebooks, is fine. However, there are exceptions: for example, if the images are extracted from a movie, then having frame *n* in the training set and frame *n+1* in the test set would result in accuracy estimates which are over-inflated since the two images are too similar.
</details>

<details>
<summary> How to design a good test set? </summary>
The test set should contain images which resemble what the input to the trained model looks like when deployed. For example, images taken under similar lighting conditions, similar angles, etc. This is to ensure that the accuracy estimate reflects the real performance of the application which uses the trained model.
</details>


<details>
<summary> How to speed up training? </summary>
- All images should be stored on an SSD device, since HDD or network access times can dominate the training time due to high latency.
- Very high-resolution images (>4 MegaPixels) should be downsized before DNN training since JPEG decoding is expensive and can slow down training by a factor of >10x.
</details>

<details>
<summary> How to improve accuracy or inference speed </summary>
See the [02_training_accuracy_vs_speed.ipynb](.notebooks/02_training_accuracy_vs_speed.ipynb) notebook for a discussion what parameters are important, and how to select a model which is fast during inference.
</details>
