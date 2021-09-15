import numpy as np
import sys
import os
import tensorflow as tf
import cv2

def ImageResize(image,factor=0.6):
	width = int(image.shape[1] * factor)
	height = int(image.shape[0] * factor)
	dim = (width, height)
	#print(image.shape)
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_LANCZOS4)
	#print(resized.shape)
	return resized

def getListOfFiles(dirName):
    print(dirName)
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        allFiles.append(entry)
    return allFiles 

def GetOverlappingBlocks(im,M=256,N=256,Part=8):
    tiles = []
    tile = np.zeros((M,N,3),dtype=np.uint8)
    #tile[:,:,2] = 255
    
    x = 0 
    y = 0
    x_start = 0
    y_start = 0
    while y < im.shape[0]:
        while x < im.shape[1]:
            if(x!=0):
                x_start = x - int(N/Part)
            if(y!=0):
                y_start = y - int(M/Part)
            if(y_start+M>im.shape[0]):
                if(x_start+N>im.shape[1]):
                    tile[0:im.shape[0]-y_start,0:im.shape[1]-x_start,:] = im[y_start:im.shape[0],x_start:im.shape[1],:]
                else:
                    tile[0:im.shape[0]-y_start,0:N,:] = im[y_start:im.shape[0],x_start:x_start+N,:]
            else:
                if(x_start+N>im.shape[1]):
                    tile[0:M,0:im.shape[1]-x_start,:] = im[y_start:y_start+M,x_start:im.shape[1],:]
                else:
                    tile[0:M,0:N,:] = im[y_start:y_start+M,x_start:x_start+N,:]
            
            
            #pre_tile = cv2.cvtColor(PreProcessInput(cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)
            #tiles.append(load_tf_img(pre_tile,M))
            
            #tiles.append(load_tf_img(tile,M))
            tiles.append(tile)

            tile = np.zeros((M,N,3),dtype=np.uint8)
            #tile[:,:,2] = 255
            x = x_start + N
        y = y_start + M
        x = 0
        x_start = 0
    return tiles


def CombineToImage(imgs,h,w,ch,Part=8):
    Image = np.zeros((h,w,ch),dtype=np.float32)
    Image_flag = np.zeros((h,w),dtype=bool)
    i = 0
    j = 0
    i_end = 0
    j_end = 0
    for k in range(len(imgs)):
        #part_img = np.copy(imgs[k,:,:,:])
        part_img = np.copy(imgs[k])
        hh,ww,cc = part_img.shape
        i_end = min(h,i + hh)
        j_end = min(w,j + ww)
        
        
        for m in range(hh):
            for n in range(ww):
                if(i+m<h):
                    if(j+n<w):
                        if(Image_flag[i+m,j+n]):
                            
                            Image[i+m,j+n,:] = (Image[i+m,j+n,:] + part_img[m,n,:])/2
                        else:
                            Image[i+m,j+n,:] = np.copy(part_img[m,n,:])

        Image_flag[i:i_end,j:j_end] = True
        j =  min(w-1,j + ww - int(ww/Part))
        #print(i,j,w)
        #print(k,len(imgs))
        if(j_end==w):
            j = 0
            i = min(h-1,i + hh - int(hh/Part))
    Image = Image*255.0
    return Image.astype(np.uint8)

def load_tf_img(img,max_dim=256):
  img = tf.convert_to_tensor(img)
  #print(img)
  img = tf.image.convert_image_dtype(img, tf.float32)
  #print(img)
  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  #print(shape)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  #print(new_shape)
  img = tf.image.resize(img, new_shape)
  return img 
