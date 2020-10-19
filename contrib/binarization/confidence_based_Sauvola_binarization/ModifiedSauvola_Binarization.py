'''
@author: Soumyadeep Dey
				email:Soumyadeep.Dey@microsoft.com
				Date: October 2020
				cite: https://drive.google.com/file/d/1D3CyI5vtodPJeZaD2UV5wdcaIMtkBbdZ/view?usp=sharing
'''

# importing required libraries
import numpy as np
import cv2
from skimage.filters import threshold_sauvola


def SauvolaModBinarization(image,n1=51,n2=51,k1=0.3,k2=0.3,default=True):
    '''
	 Binarization using Sauvola's algorithm
		@name : SauvolaModBinarization
	 parameters
		@param image (numpy array of shape (3/1) of type np.uint8): color or gray scale image
	 optional parameters
		@param n1 (int) : window size for running sauvola during the first pass
		@param n2 (int): window size for running sauvola during the second pass
		@param k1 (float): k value corresponding to sauvola during the first pass
		@param k2 (float): k value corresponding to sauvola during the second pass
		@param default (bool) : bollean variable to set the above parameter as default. 

			@param default is set to True : thus default values of the above optional parameters (n1,n2,k1,k2) are set to
				n1 = 5 % of min(image height, image width)
				n2 = 10 % of min(image height, image width)
				k1 = 0.5
				k2 = 0.5
		Returns
			@return A binary image of same size as @param image
		
		@cite https://drive.google.com/file/d/1D3CyI5vtodPJeZaD2UV5wdcaIMtkBbdZ/view?usp=sharing
    '''

    if(default):
        n1 = int(0.05*min(image.shape[0],image.shape[1]))
        if (n1%2==0):
            n1 = n1+1
        n2 = int(0.1*min(image.shape[0],image.shape[1]))
        if (n2%2==0):
            n2 = n2+1
        k1 = 0.5
        k2 = 0.5
    if(image.ndim==3):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = np.copy(image)
    T1 = threshold_sauvola(gray, window_size=n1,k=k1)
    max_val = np.amax(gray)
    min_val = np.amin(gray)
    C = np.copy(T1)
    C = C.astype(np.float32)
    C[gray > T1] = (gray[gray > T1] - T1[gray > T1])/(max_val - T1[gray > T1])
    C[gray <= T1] = 0
    C = C * 255.0
    new_in = np.copy(C.astype(np.uint8))
    T2 = threshold_sauvola(new_in, window_size=n2,k=k2)
    binary = np.copy(gray)
    binary[new_in <= T2] = 0
    binary[new_in > T2] = 255
    return binary
