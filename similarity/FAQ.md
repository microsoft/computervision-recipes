# Image similarity

## Frequently asked questions


* General
  * [Which problems can be solved using image similarity, and which ones cannot?](#which-problems-can-be-solved-using-image-similarity)
  * [When should I use image similarity and not another method?](#when-should-i-use-image-similarity-and-not-another-method)

### Which problems can be solved using image similarity?
Image similarity is typically used to build Image Retrieval systems where, given a *query* image, the goal is to find all similar images in a *reference* set. These systems can be used to help users find products comparable to what they like, but cheaper, closer to their location or even alternatives to items that are currently out of stock.

[00_webcam.ipynb](notebooks/00_webcam.ipynb) notebook shows an example implementation of such a system.

### When should I use image similarity and not another method?
Image similarity can be used if the object-of-interest is relatively large in the image, e.g. more than 20% image width/height. If the object is smaller, then object detection methods should be used in a pre-processing step to locate and crop the image to the area of interest.
