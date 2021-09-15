import tensorflow as tf
import numpy as np
import cv2
import os
import sys

from train import train

data_folder = 'sample_data'
gt_folder = 'sample_gt_data'
batch_size = 21

train(data_folder,gt_folder,dataset_path='dataset',checkpoint='checkpoints',train_batch_size=batch_size)
