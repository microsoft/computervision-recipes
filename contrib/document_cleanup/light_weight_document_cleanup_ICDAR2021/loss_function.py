import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda
import tensorflow.keras.backend as K
import os
import sys
import numpy as np





def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

'''
# Content layer where will pull our feature maps
content_layers = ['block2_conv2'] 

# Style layer of interest
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)
'''

def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]
  

  model = tf.keras.Model([vgg.input], outputs)
  return model

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    #tf.keras.backend.clear_session()	  
    super(StyleContentModel, self).__init__()
    self.vgg =  vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.num_content_layers = len(content_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers], 
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name:value 
                    for content_name, value 
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name:value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}
    
    return {'content':content_dict, 'style':style_dict}
    
def Compute_PLoss(in_img,gt_img,style_weight,content_weight):
	#tf.keras.backend.clear_session()
	preprocessed_in = tf.keras.applications.vgg19.preprocess_input(in_img*255)
	preprocessed_gt = tf.keras.applications.vgg19.preprocess_input(gt_img*255)
	# Content layer where will pull our feature maps
	content_layers = ['block2_conv2'] 

	# Style layer of interest
	style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']
	extractor = StyleContentModel(style_layers, content_layers)
	in_out = extractor(preprocessed_in)
	gt_out = extractor(preprocessed_gt)
	style_outputs_in = in_out['style']
	content_outputs_in = in_out['content']
	style_outputs_gt = gt_out['style']
	content_outputs_gt = gt_out['content']
	style_loss = tf.add_n([tf.reduce_mean(abs(style_outputs_in[name]-style_outputs_gt[name])) for name in style_outputs_in.keys()])
	style_loss *= style_weight / extractor.num_style_layers
	content_loss = tf.add_n([tf.reduce_mean(abs(content_outputs_in[name]-content_outputs_gt[name])) for name in content_outputs_in.keys()])   
	content_loss *= content_weight / extractor.num_content_layers
	PLoss = tf.math.add_n([style_loss,content_loss])
	return PLoss

def IlluminationLoss(y_gt,y_out,gray_flag=True,style_weight=1e-2,content_weight=1e2):
    
    #tf.keras.backend.clear_session()

    #####################################################################
    if(gray_flag):
        rgb_out = tf.image.grayscale_to_rgb(y_out)
        PLoss = Compute_PLoss(rgb_out,y_gt,style_weight,content_weight)
        #print('Ploss',PLoss)
        gray_gt = tf.image.rgb_to_grayscale(y_gt)
        gray_loss = tf.reduce_mean(abs(gray_gt - y_out)) #loss in gray space
        gray_loss = tf.math.scalar_mul(1e2,gray_loss)
        #print('gray loss', gray_loss)
        loss = tf.math.add_n([PLoss,gray_loss])
        #print('loss is', loss)
        return loss
    #####################################################################
    PLoss = Compute_PLoss(y_out,y_gt,style_weight,content_weight)
    #print('loss is', loss)
    #RGB Loss
    rgb_loss = tf.reduce_mean(abs(y_gt[:,:,:,0] - y_out[:,:,:,0])) + tf.reduce_mean(abs(y_gt[:,:,:,1] - y_out[:,:,:,1])) + tf.reduce_mean(abs(y_gt[:,:,:,2] - y_out[:,:,:,2]))#loss in RGB color space
    rgb_loss = tf.math.scalar_mul(1e2,rgb_loss)
    #print('######################')
    #print('rgb_loss',rgb_loss)
    #print('######################')
    #####################################################################
    #Color Loss (Hue)
    hsv_out = tf.image.rgb_to_hsv(y_out)
    hsv_gt = tf.image.rgb_to_hsv(y_gt)
    #hsv_loss = tf.reduce_mean(min(min(hsv_gt[:,:,:,0],hsv_out[:,:,:,0])*360+(360-max(hsv_gt[:,:,:,0],hsv_out[:,:,:,0])*360),abs(hsv_gt[:,:,:,0] - hsv_out[:,:,:,0])*360))
    hue_loss = tf.reduce_mean(abs(hsv_gt[:,:,:,0] - hsv_out[:,:,:,0])) #loss in hue color space
    hue_loss = tf.math.scalar_mul(1e2,hue_loss)
    #print('hue_loss',hue_loss)
    
    yuv_out = tf.image.rgb_to_yuv(y_out)
    yuv_gt = tf.image.rgb_to_yuv(y_gt)
    y_loss = tf.reduce_mean(abs(yuv_gt[:,:,:,0] - yuv_out[:,:,:,0])) #loss in luminance
    y_loss = tf.math.scalar_mul(1e2,y_loss)
    #print('luminance loss', y_loss)
    
    #####################################################################
    loss = tf.math.add_n([PLoss,rgb_loss])
    #loss = tf.math.add_n([PLoss,rgb_loss,y_loss])
    #loss = tf.math.add_n([PLoss,rgb_loss,hue_loss])
    #print('######################')
    #print('loss is', loss)
    #print('######################')
    
    return loss

def illu_Loss(style_weight,content_weight,gray_flag):
    def ILoss(y_gt, y_out):
        return IlluminationLoss(y_gt,y_out,gray_flag,style_weight,content_weight)
    return ILoss
