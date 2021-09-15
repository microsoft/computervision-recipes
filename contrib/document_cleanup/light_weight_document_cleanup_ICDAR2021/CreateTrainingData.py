import cv2
import os
import random
import numpy as np
from random import randint
import albumentations as A
import numpy as np
import cv2
import sys
import os
from tqdm import tqdm
from utils import GetOverlappingBlocks, getListOfFiles, ImageResize



transform = A.Compose([
        A.OneOf([
            A.ISONoise(p=0.4),
            A.JpegCompression(quality_lower=50, quality_upper=70, always_apply=False, p=0.8),
        ], p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=10,p=.8),
            A.MedianBlur(blur_limit=3, p=0.75),
            A.GaussianBlur(blur_limit=7, p=0.75),
        ], p=0.8),                       
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3,p=0.75),         
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=18, shadow_dimension=6, p=0.85),
        ], p=0.8),
    ])



def GenerateTrainingBlocks(data_folder,gt_folder,dataset_path='./dataset',M=256,N=256):
	print(data_folder)
	print('Generating training blocks!!!')
	train_path =  dataset_path + '/' + data_folder + '_Trainblocks'

	if not os.path.exists(train_path):
		os.makedirs(train_path)


	train_filenames = train_path + '/train_block_names.txt'
	f = open(train_filenames, 'w')
	
	data_path = dataset_path + '/' + data_folder
	gt_path = dataset_path + '/' + gt_folder
	
	print(data_path)
	
	filenames =  getListOfFiles(data_path)
	cnt = 0
	print(filenames)
	for name in tqdm(filenames):
		print(name)
		gt_filename = gt_path + '/' + name
		in_filename = data_path + '/' + name
		print(gt_filename)
		print(in_filename)
		gt_image_initial = cv2.imread(gt_filename)
		in_image_initial = cv2.imread(in_filename)
		print(gt_image_initial.shape,in_image_initial.shape)
		for scale in [0.7,1.0,1.4]: 
			gt_image = ImageResize(gt_image_initial, scale)
			in_image = ImageResize(in_image_initial, scale)	
			h,w,c = in_image.shape
			gt_img = GetOverlappingBlocks(gt_image,Part=8)
			in_img = GetOverlappingBlocks(in_image,Part=8)
			for i in range(len(gt_img)):           
				train_img_path = train_path + '/block_' + str(cnt) + '.png'
				gt_img_path = train_path + '/gtblock_' + str(cnt) + '.png'
				cv2.imwrite(train_img_path,in_img[i])
				#cv2.imwrite(train_img_path,PreProcessInput(in_img[i]))
				cv2.imwrite(gt_img_path,gt_img[i])
				t_name = 'block_' + str(cnt) + '.png'
				f.write(t_name)
				f.write('\n')
				cnt += 1
			Random_Block_Number_PerImage = int(len(gt_img)/5) 
			for i in range(Random_Block_Number_PerImage):
				
				if(in_image.shape[0]-M>1 and in_image.shape[1]-N>1):
					y = random.randint(1, in_image.shape[0]-M)  
					x = random.randint(1, in_image.shape[1]-N)
					in_part_img = in_image[y:y+M,x:x+N,:].copy() 
					gt_part_img = gt_image[y:y+M,x:x+N,:].copy()
					train_img_path = train_path + '/block_' + str(cnt) + '.png'
					gt_img_path = train_path + '/gtblock_' + str(cnt) + '.png'
					in_part_img = cv2.cvtColor(in_part_img, cv2.COLOR_BGR2RGB)
					augmented_image = transform(image=in_part_img)['image']
					augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
					
					cv2.imwrite(train_img_path,augmented_image)
					cv2.imwrite(gt_img_path,gt_part_img)	
					t_name = 'block_' + str(cnt) + '.png'
					f.write(t_name)
					f.write('\n')
					cnt += 1
				else:
					break
					in_part_img = np.zeros((M,N,3),dtype=np.uint8)
					gt_part_img = np.zeros((M,N,3),dtype=np.uint8)
					in_part_img[:,:,:] = 255
					gt_part_img[:,:,:] = 255
					
					if(in_image.shape[0]-M<=1 and in_image.shape[1]-N>1):
						y = 0
						x = random.randint(1, in_image.shape[1]-N)
						in_part_img[:h,:,:] = in_image[:,x:x+N,:].copy() 
						gt_part_img[:h,:,:] = gt_image[:,x:x+N,:].copy()
					if(in_image.shape[0]-M>1 and in_image.shape[1]-N<=1):
						x = 0
						y = random.randint(1, in_image.shape[0]-M)  
						in_part_img[:,:w,:] = in_image[y:y+M,:,:].copy() 
						gt_part_img[:,:w,:] = gt_image[y:y+M,:,:].copy()	
					
					
					train_img_path = train_path + '/block_' + str(cnt) + '.png'
					gt_img_path = train_path + '/gtblock_' + str(cnt) + '.png'
					in_part_img = cv2.cvtColor(in_part_img, cv2.COLOR_BGR2RGB)
					augmented_image = transform(image=in_part_img)['image']
					augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
					
					cv2.imwrite(train_img_path,augmented_image)
					cv2.imwrite(gt_img_path,gt_part_img)	
					t_name = 'block_' + str(cnt) + '.png'
					f.write(t_name)
					f.write('\n')
					cnt += 1
		#print(cnt)
 
            
	f.close()

	print('Total number of training blocks generated: ', cnt)
	
	return train_path, train_filenames

	
