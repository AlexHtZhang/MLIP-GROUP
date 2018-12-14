import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Flatten
from keras import optimizers
from keras.models import Model
from keras import backend as K
import numpy

def classifier(n):
	# load pima indians dataset
	dataset = numpy.load("train.npy") # 18643 * 32768
	label = numpy.load("train_one_hot.npy") # 18643 * 28
	# split into input (X) and output (Y) variables
	X = dataset
	X = numpy.reshape(X,(X.shape[0],64,64,8))
	Y = label[:,n]
	# create model
	input_img = Input(shape=(64, 64, 8))
	x = Convolution2D(4, (1, 1), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Convolution2D(4, (1, 1), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Convolution2D(2, (1, 1), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Convolution2D(2, (1, 1), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	fc= Flatten()(x)
	fc= Dense(64, activation='relu')(fc)
	fc= Dropout(0.15)(fc)
	fc= Dense(32, activation='relu')(fc)
	fc= Dropout(0.15)(fc)
	fc= Dense(16, activation='relu')(fc)
	fc= Dropout(0.15)(fc)
	fc= Dense(8, activation='relu')(fc)
	fc= Dropout(0.15)(fc)
	fc= Dense(4, activation='relu')(fc)
	fc= Dropout(0.15)(fc)
	fc= Dense(2, activation='relu')(fc)
	output= Dense(1, activation='sigmoid')(fc)
	model=Model(input_img, output)
	# Compile model
	sgd = optimizers.SGD(lr=0.01, momentum=0.9,nesterov=True)
	class_weight = {0: 0.4*float((Y==0).sum()),1: float((Y==1).sum())}
	model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# Fit the model
	model.fit(X, Y, epochs=5, batch_size=128,class_weight=class_weight)
	return model

for n in range(28):
	classifier(n).save('model_'+str(n)+'.h5')
	
#classifier(0).save('model_'+str(0)+'.h5')