__author__ = "Chenghao_Gong"
__copyright__ = "Copyright 2018, MLIP Project"
__license__ = "MIT"
__version__ = "1.0"
__status__ = "Beta"

import os
import sys
import numpy as np
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
import cv2
def load_filenames(data_path,image_names_path):
	#load train image names
	file=open(image_names_path+'train_lables.txt','r')
	train=[]
	for line in file:
		train.append(line[:len(line)-1])
	file.close()
	#load test image names
	file=open(image_names_path+'test_lables.txt','r')
	test=[]
	for line in file:
		test.append(line[:len(line)-1])
	file.close()
	#load val image names
	file=open(image_names_path+'validation_lables.txt','r')
	val=[]
	for line in file:
		val.append(line[:len(line)-1])
	file.close()
	train_all=train+test+val
	return train,test,val
def load_batch(train_all,data_path,batch_size,which_batch):
	image_list=[]
	for i in range(which_batch*batch_size,(which_batch+1)*batch_size):
		image=[]
		image.append(cv2.imread(data_path+train_all[i]+'_red.png',0))
		image.append(cv2.imread(data_path+train_all[i]+'_green.png',0))
		image.append(cv2.imread(data_path+train_all[i]+'_blue.png',0))
		image.append(cv2.imread(data_path+train_all[i]+'_yellow.png',0))
		image=np.asarray(image).T
		image_list.append(image)
	image_list=np.asarray(image_list)
	return image_list
#load_batch(train_all,32,0)
def build_model():
	input_img = Input(shape=(512, 512, 4)) # 1ch=black&white, 28 x 28
	x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img) #nb_filter, nb_row, nb_col
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)
	print ("shape of encoded")
	print (K.int_shape(encoded))

	x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x) 
	x = UpSampling2D((2, 2))(x)
	decoded = Convolution2D(4, (5, 5), activation='sigmoid', padding='same')(x)
	print ("shape of decoded")
	print (K.int_shape(decoded))
	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return autoencoder
def normalize(image_list):
	ma=max(image_list.flatten())
	mi=min(image_list.flatten())
	output=image_list/(ma-mi)-mi/(ma-mi)
	return output