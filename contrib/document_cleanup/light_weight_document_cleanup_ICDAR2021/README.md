# Document Image Cleanup
Given an input noisy document image, the aim of document image cleanup is to improve its readability and visibility by removing the noisy elements.

## Light-weight Document Image Cleanup using Perceptual Loss

Smartphones have enabled effortless capturing and sharing of documents in digital form. The documents, however, often undergo various types of degradation due to aging, stains, or shortcoming of capturing environment such as shadow, non-uniform lighting, etc., which reduces the comprehensibility of the document images. In this work, we consider the problem of document image cleanup on embedded applications such as smartphone apps, which usually have memory, energy, and latency limitations due to the device and/or for best human user experience. We propose a light-weight encoder decoder based convolutional neural network architecture for removing the noisy elements from document images. To compensate for generalization performance with a low network capacity, we incorporate the perceptual loss for knowledge transfer from pre-trained deep CNN network in our loss function. In terms of the number of parameters and product-sum operations, our models are 65-1030 and 3-27 times, respectively, smaller than existing state-of-the-art document enhancement models. Overall, the proposed models offer a favorable resource versus accuracy trade-off and we empirically illustrate the efficacy of our approach on several real-world benchmark datasets.

### cite

https://link.springer.com/chapter/10.1007/978-3-030-86334-0_16

@InProceedings{10.1007/978-3-030-86334-0_16, author="Dey, Soumyadeep and Jawanpuria, Pratik", editor="Llad{'o}s, Josep and Lopresti, Daniel and Uchida, Seiichi", title="Light-Weight Document Image Cleanup Using Perceptual Loss", booktitle="Document Analysis and Recognition -- ICDAR 2021", year="2021", publisher="Springer International Publishing", address="Cham", pages="238--253", isbn="978-3-030-86334-0" }


### Noisy input images

<img src="./sample_input_output/book_org.jpg" width="33%"> </img>
<img src="./sample_input_output/pres1_org.jpg" width="33%"> </img>
<img src="./sample_input_output/bill_org.jpg" width="33%"> </img>

### cleanup images

<img src="./sample_input_output/book_dnn.jpg" width="33%"> </img>
<img src="./sample_input_output/pres1_dnn.jpg" width="33%"> </img>
<img src="./sample_input_output/bill_dnn.jpg" width="33%"> </img>

## Setup

### Dependencies
- python 3.7
- numpy 1.16
- opencv 4.2
- skimage 0.17
- tensorflow 2.4
- albumentations
- tqdm
- scikit-learn


### Example

Sample example of the usage (training and testing) of the proposed cleanup technique [notebook](./DocumentCleanup_ICDAR2021.ipynb).
