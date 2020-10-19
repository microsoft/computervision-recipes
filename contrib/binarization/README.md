# Binarization
Binarization is a technique to segment foreground from the background pixels. A simple technique for binarization is thresholding of gray-level or color document scanned images.
## At a glance

This binarization technique is an improvement over Sauvola's binarization technique. In this work, we improve the existing Sauvola's binarization technique by preserving more foreground information in the binarized document-images. In order to achieve this, we introduce a confidence score for the background pixels. 

### Input images

<img src="./confidence_based_Sauvola_binarization/test_images/2.jpeg" width="33%"> </img>
<img src="./confidence_based_Sauvola_binarization/test_images/10.jpeg" width="33%"> </img>
<img src="./confidence_based_Sauvola_binarization/test_images/new1.jpg" width="33%"> </img>

### Binary outputs

<img src="./confidence_based_Sauvola_binarization/results/2_bin_new.png" width="33%"> </img>
<img src="./confidence_based_Sauvola_binarization/results/10_bin_new.png" width="33%"> </img>
<img src="./confidence_based_Sauvola_binarization/results/new1_bin_new.png" width="33%"> </img>
