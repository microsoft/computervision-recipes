import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda
from sklearn.model_selection import train_test_split
import os
import sys
import cv2
from datetime import datetime
import numpy as np

from CreateTrainingData import GenerateTrainingBlocks
from model import GetModel
from utils import load_tf_img
from loss_function import IlluminationLoss, illu_Loss

#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
os.environ["CUDA_VISIBLE_DEVICES"]='0'

#gpu_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpu_devices[0], True)

#tf.config.experimental.set_memory_growth(gpu_devices[0], True)
##tf.config.experimental.set_memory_growth(gpu_devices[1], True)
##tf.config.experimental.set_memory_growth(gpu_devices[2], True)
##tf.config.experimental.set_memory_growth(gpu_devices[3], True)

#mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:2", "/gpu:3"])
#mirrored_strategy = tf.distribute.MirroredStrategy()

#tf.config.run_functions_eagerly(True)
tf.config.experimental_run_functions_eagerly(True)

def GetTrainFileNames(file_name_path):
    train_image_names = []
    with open(file_name_path) as fp:
        for line in fp:
            filename = line.rstrip()
            name = filename
            train_image_names.append(name)

    return train_image_names


def ImageResizeSquare(image):
        if(image.shape[1]!=image.shape[0]):
                width = max(image.shape[1],image.shape[0])
                height = width
                dim = (width, height)
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_LANCZOS4)
                return resized
        else:
                return image

def GetData(filenames,path,block_size=(256,256)):
    max_d = max(block_size[0],block_size[1])
    gt_imgs = []
    in_imgs = []
    cnt = 0
    for name in filenames:
        #print(cnt,len(filenames))
        gt_filename = path + '/gt' + name
        in_filename = path + '/' + name
        gt_image = load_tf_img(cv2.imread(gt_filename,1),max_d)
        in_image = load_tf_img(cv2.imread(in_filename,1),max_d)
        in_imgs.append(in_image)
        gt_imgs.append(gt_image)
        cnt += 1
    return in_imgs, gt_imgs
                
class My_Custom_Generator(tf.keras.utils.Sequence) :

  def __init__(self, image_filenames, img_dir, batch_size) :
    self.image_filenames = image_filenames
    self.batch_size = batch_size
    self.img_dir = img_dir


  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)


  def __getitem__(self, idx) :
    batch_x = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    gt_imgs = []
    in_imgs = []
    for name in batch_x:
        #print(name,idx)
        gt_filename = self.img_dir + '/gt' + name
        in_filename = self.img_dir + '/' + name
        gt_image = load_tf_img(ImageResizeSquare(cv2.cvtColor(cv2.imread(gt_filename,1), cv2.COLOR_BGR2RGB)))
        in_image = load_tf_img(ImageResizeSquare(cv2.cvtColor(cv2.imread(in_filename,1), cv2.COLOR_BGR2RGB)))
        in_imgs.append(in_image)
        gt_imgs.append(gt_image)
    return tf.convert_to_tensor(in_imgs), tf.convert_to_tensor(gt_imgs)

def train(data_folder,gt_folder,dataset_path='dataset',checkpoint='checkpoints',epochs=10,pretrain_flag=False,pretrain_model_weight_path=None,model_name='M32',gray_flag=True,block_size=(256,256),train_batch_size=1):
	block_height = block_size[0]
	block_width = block_size[1]
	print(block_height,block_width)
	print(data_folder)
	print(gt_folder)
	
	train_path, train_filenames = GenerateTrainingBlocks(data_folder=data_folder,gt_folder=gt_folder,dataset_path=dataset_path,M=block_height ,N=block_width)
	train_image_names = GetTrainFileNames(train_filenames)
	X_train_filenames, X_val_filenames, y_train, y_val = train_test_split(
            train_image_names, train_image_names, test_size=0.2, random_state=1)
	
	my_training_batch_generator = My_Custom_Generator(X_train_filenames, train_path, train_batch_size)
	my_validation_batch_generator = My_Custom_Generator(X_val_filenames, train_path, train_batch_size)
	in_imgs, gt_imgs = GetData(train_image_names[:2],train_path)
	#print(IlluminationLoss(gt_imgs[0][tf.newaxis, :],in_imgs[0][tf.newaxis, :],style_weight=1e-1,content_weight=1e1,gray_flag=gray_flag))

	if(gray_flag):
		print(IlluminationLoss(gt_imgs[0][tf.newaxis, :],tf.image.rgb_to_grayscale(in_imgs[0][tf.newaxis, :]),style_weight=1e-1,content_weight=1e1,gray_flag=gray_flag))
	else:
		print(IlluminationLoss(gt_imgs[0][tf.newaxis, :],in_imgs[0][tf.newaxis, :],style_weight=1e-1,content_weight=1e1,gray_flag=gray_flag ))
	
	#logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = os.path.join('logs','scalars')
	logdir = os.path.join(logdir,datetime.now().strftime("%Y%m%d-%H%M%S"))
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	
	
	custom_loss = illu_Loss(style_weight=1e-1,content_weight=1e1,gray_flag=gray_flag)          
	model = GetModel(model_name=model_name,gray=gray_flag,block_size=block_size)
	opt = tf.keras.optimizers.Adam()
	model.compile(optimizer=opt, loss = custom_loss)
	
	#Initialize model with pre-trained weights
	if(pretrain_flag):
		model.load_weights(pretrain_model_weight_path)
	
	#Saving Model file to checkpoint folder
	Illumodel_json = model.to_json()
	model_name_suffix = ''
	model_weight_suffix = ''
	if(gray_flag):
		model_name_suffix = '_gray.json'
		model_weight_suffix = '_gray'
	else:
		model_name_suffix = '_color.json'
		model_weight_suffix = '_color'
	save_model_name = model_name + model_name_suffix
	#print(checkpoint)
	save_model_path = os.path.join(checkpoint, save_model_name)
	print(save_model_path)
	with open(save_model_path, "w") as json_file:
		json_file.write(Illumodel_json)
		json_file.close()
	# checkpoint
	model_weight_name = model_name + model_weight_suffix + '_' + data_folder +'_epoch-{epoch:02d}.hdf5'
	full_model_weight_path = os.path.join(checkpoint, model_weight_name)
	#print(filepath)
	model_checkpoint = tf.keras.callbacks.ModelCheckpoint(full_model_weight_path, monitor='val_loss', verbose=1, save_best_only=True)
	
	callbacks_list = [tensorboard_callback, model_checkpoint]
	
	training_history = model.fit(my_training_batch_generator,
											epochs = epochs, verbose=1, workers = 21, use_multiprocessing = False,
											validation_data = my_validation_batch_generator,     
											callbacks=callbacks_list)
	
