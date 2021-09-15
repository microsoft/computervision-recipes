import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda

import os
import sys

def res_net_block(input_data, filters, conv_size):
    x = layers.Conv2D(filters, conv_size, activation=tf.nn.relu6, padding='same',kernel_initializer = 'he_normal')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, conv_size, activation=tf.nn.relu6, padding='same',kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, input_data])
    x = layers.Activation(tf.nn.relu6)(x)
    return x

def convert2gray(in_tensor):
    out = tf.image.rgb_to_grayscale(in_tensor)
    return out

def CreateModel_M16_binary(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    gray_in = layers.Lambda(lambda x : convert2gray(x))(X_input)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(1, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([gray_in, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='M16Gray')
    return model

def CreateModel_M16_color(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = res_net_block(out, 16, 3)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(3, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([X_input, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='M16Color')
    return model


def CreateModel_M32_binary(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    gray_in = layers.Lambda(lambda x : convert2gray(x))(X_input)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut2 = out
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = layers.add([shortcut2, out])
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(1, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([gray_in, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='IlluNet')

    return model

def CreateModel_M32_color(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut2 = out
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = res_net_block(out, 32, 3)
    out = layers.add([shortcut2, out])
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(3, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([X_input, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='IlluNet')

    return model


def CreateModel_M64_binary(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    gray_in = layers.Lambda(lambda x : convert2gray(x))(X_input)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut2 = out
    out = layers.Conv2D(64, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut1 = out
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = layers.add([shortcut1, out])
    out = layers.Conv2D(64, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut2, out])
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(1, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([gray_in, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='IlluNet')

    return model

def CreateModel_M64_color(input_shape = (None, None, 3),batch_size=None):
    _strides=(1, 1)
    # Define the input as a tensor with shape input_shape
    X_input = Input(shape=input_shape,batch_size=batch_size)
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(X_input)
    out = layers.BatchNormalization()(out)
    shortcut3 = out
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut2 = out
    out = layers.Conv2D(64, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    shortcut1 = out
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = res_net_block(out, 64, 3)
    out = layers.add([shortcut1, out])
    out = layers.Conv2D(64, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut2, out])
    out = layers.Conv2D(16, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)
    out = layers.BatchNormalization()(out)
    out = layers.add([shortcut3, out])
    out = layers.Conv2D(3, kernel_size=(3, 3),activation=tf.nn.relu6, strides=_strides, padding='same',kernel_initializer = 'he_normal')(out)   
    out = layers.BatchNormalization()(out)
    out = layers.add([X_input, out])
    out = tf.math.sigmoid(out)
    # Create model
    model = Model(inputs = X_input, outputs = out, name='IlluNet')

    return model



def GetModel(model_name='M32',gray=True,block_size=(None,None),batch_size=None):
	input_shape = (block_size[0],block_size[1],3)
	if(model_name=='M64'):
		if(gray):
			return CreateModel_M64_binary(input_shape,batch_size)
		else:
			return CreateModel_M64_color(input_shape,batch_size)
	elif(model_name=='M32'):
		if(gray):
			return CreateModel_M32_binary(input_shape,batch_size)
		else:
			return CreateModel_M32_color(input_shape,batch_size)
	else:
		if(gray):
			return CreateModel_M16_binary(input_shape,batch_size)
		else:
			return CreateModel_M16_color(input_shape,batch_size)
	
