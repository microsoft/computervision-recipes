# Image similarity

## Frequently asked questions


* General
  * [Which problems can be solved using image similarity, and which ones cannot?](#which-problems-can-be-solved-using-image-similarity)
  * [When should I use image similarity and not another method?](#when-should-i-use-image-similarity-and-not-another-method)

### Which problems can be solved using image similarity?
Image similarity is often used to build Image Retrieval systems where, given a *query* image, the goal is to find all similar images in a *reference* set. These systems can be used e.g. on a shopping website to suggest users comparable products, or to cluster large corporate image datasets into groups with similar content and appearance.

### When should I use image similarity and not another method?
Image similarity can be used if the object-of-interest is relatively large in the image, e.g. more than 20% image width/height. If the object is smaller, then object detection methods should be used in a pre-processing step to locate and crop the image to the area of interest.
