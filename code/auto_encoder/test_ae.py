import os
import sys
import cnn_ae as ae 
import numpy as np
from keras.models import load_model
from keras.models import Model
import matplotlib.pyplot as plt

data_path=os.getcwd()+'/../../train/'# Change this to the path to the image folder
image_names_path=os.getcwd()+'/../processed_data/'
train,test,val=ae.load_filenames(data_path,image_names_path)
train_all=train+test+val
model=load_model('auto_encoder.h5')
#image=ae.load_batch(train_all,data_path,1,0)
#image=ae.normalize(image)
#print(image.shape)
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('max_pooling2d_3').output)
#print(intermediate_layer_model.predict(image).shape)
train_output=[]
for i in range(len(train)):
	print(i)
	image=ae.load_batch(train,data_path,1,i)
	image=ae.normalize(image)
	output=intermediate_layer_model.predict(image)
	train_output.append(output.flatten())
train_output=np.asarray(train_output)
np.save('train.npy', train_output)

test_output=[]
for i in range(len(test)):
	print(i)
	image=ae.load_batch(test,data_path,1,i)
	image=ae.normalize(image)
	output=intermediate_layer_model.predict(image)
	test_output.append(output.flatten())
test_output=np.asarray(test_output)
np.save('test.npy', test_output)

val_output=[]
for i in range(len(val)):
	print(i)
	image=ae.load_batch(val,data_path,1,i)
	image=ae.normalize(image)
	output=intermediate_layer_model.predict(image)
	val_output.append(output.flatten())
val_output=np.asarray(val_output)
np.save('val.npy', val_output)

# plotting code
image=ae.load_batch(train_all,data_path,1,0)
image=ae.normalize(image)
print(image.shape)
image_=model.predict(image)
print(image_.shape)
plt.subplot(221)
plt.title('Original Image Red Channel')
plt.imshow(image[0,:,:,0])
plt.subplot(222)
plt.title('Original Image Green Channel')
plt.imshow(image[0,:,:,1])
plt.subplot(223)
plt.title('Original Image Blue Channel')
plt.imshow(image[0,:,:,2])
plt.subplot(224)
plt.title('Original Image Yellow Channel')
plt.imshow(image[0,:,:,3])
plt.show()

plt.subplot(221)
plt.title('Reconstructed Image Red Channel')
plt.imshow(image_[0,:,:,0])
plt.subplot(222)
plt.title('Reconstructed Image Green Channel')
plt.imshow(image_[0,:,:,1])
plt.subplot(223)
plt.title('Reconstructed Image Blue Channel')
plt.imshow(image_[0,:,:,2])
plt.subplot(224)
plt.title('Reconstructed Image Yellow Channel')
plt.imshow(image_[0,:,:,3])
plt.show()
