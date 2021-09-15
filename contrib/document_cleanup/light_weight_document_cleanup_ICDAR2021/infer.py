import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model

from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D

import os
from model import convert2gray
from utils import GetOverlappingBlocks, CombineToImage,load_tf_img,getListOfFiles
from tqdm import tqdm
import cv2
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]= '0'

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)




def prepare_data_blocks(blocks,size):
	data = []
	for block in blocks:
		data.append(load_tf_img(block,size))
	#blocks = []
	return data
	

def infer(model_name,model_weight,target_dir,save_out_dir,block_size=(256,256),batch_size=1):
	json_file = open(model_name, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json,custom_objects={'relu6': tf.nn.relu6, 'convert2gray': convert2gray})


	model.summary()
	#exit(0)

	model.compile(optimizer='adam', loss = 'mean_squared_error')

	model.load_weights(model_weight)
	
	if not os.path.exists(save_out_dir):
		os.makedirs(save_out_dir)
	
	M = block_size[0]
	N = block_size[1]	
	part = 8
	filelists = getListOfFiles(target_dir)
	for filename in tqdm(filelists):
		initial_filename = os.path.splitext(filename)[0]
		in1_filename = os.path.join(target_dir,filename) 
		in_clr = cv2.imread(in1_filename,1)
		in1_image = cv2.cvtColor(in_clr, cv2.COLOR_BGR2RGB)
		in1_img = GetOverlappingBlocks(in1_image.copy(),M,N,part)
		prepared_data_blocks = prepare_data_blocks(in1_img,M)
		in1_img = []
		out_img1 = model.predict(tf.convert_to_tensor(prepared_data_blocks), batch_size=batch_size)
		num_img,ht,wd,ch_out = out_img1.shape
		h,w,ch = in_clr.shape
		if(ch_out>1):
			c_image = cv2.cvtColor(CombineToImage(out_img1,h,w,ch_out), cv2.COLOR_RGB2BGR,part)
			out_image_name = initial_filename + '.png'
			name_fig = os.path.join(save_out_dir, out_image_name)
			cv2.imwrite(name_fig,c_image)
		else:
			c_image = CombineToImage(out_img1,h,w,ch_out,part)
			out_image_name = initial_filename + '.png'
			name_fig = os.path.join(save_out_dir, out_image_name)
			cv2.imwrite(name_fig,c_image)

def infer_image(model_name,model_weight,target_image,out_image_name,block_size=(256,256),batch_size=1):
	json_file = open(model_name, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json,custom_objects={'relu6': tf.nn.relu6})
	#model = model_from_json(loaded_model_json,custom_objects={'HeNormal':tf.keras.initializers.he_normal(),'relu6': tf.nn.relu6, 'convert2gray': convert2gray,'Functional':tf.keras.models.Model})


	model.summary()
	#exit(0)

	model.compile(optimizer='adam', loss = 'mean_squared_error')

	model.load_weights(model_weight)
	
	#if not os.path.exists(save_out_dir):
	#	os.makedirs(save_out_dir)
	
	M = block_size[0]
	N = block_size[1]	
	#print(M,N)
	part = 8
	in_clr = cv2.imread(target_image,1)
	in1_image = cv2.cvtColor(in_clr, cv2.COLOR_BGR2RGB)
	in1_img = GetOverlappingBlocks(in1_image.copy(),M,N,part)
	#print(len(in1_img))
	prepared_data_blocks = prepare_data_blocks(in1_img,M)
	in1_img = []
	#prepared_data_blocks = NewGetOverlappingBlocks(in_clr.copy(),M,N,part)
	
	out_img1 = model.predict(tf.convert_to_tensor(prepared_data_blocks), batch_size=batch_size)
	
	num_img,ht,wd,ch_out = out_img1.shape
	h,w,ch = in_clr.shape
	#print(num_img)

	if(ch_out>1):
		c_image = cv2.cvtColor(CombineToImage(out_img1,h,w,ch_out), cv2.COLOR_RGB2BGR,part)
		cv2.imwrite(out_image_name,c_image)
	else:
		c_image = CombineToImage(out_img1,h,w,ch_out,part)
		cv2.imwrite(out_image_name,c_image)
