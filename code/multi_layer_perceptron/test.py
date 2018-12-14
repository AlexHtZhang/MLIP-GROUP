import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras import models
from keras.layers import Dense
import numpy as np
from sklearn.metrics import confusion_matrix

f1 = open('accuracy.txt', 'w')
f2 = open('confusion_matrix.txt', 'w')
data = np.load("test.npy") 
data = np.reshape(data,(data.shape[0],64,64,8))
label = np.load("test_one_hot.npy") 
for n in range(28):
	model = models.load_model('./Models/model_'+str(n)+'.h5')
	f1.write(np.array2string(np.array(model.evaluate(data, label[:,n]))) + '\n')
	f2.write(np.array2string(np.array(confusion_matrix(label[:,n], model.predict(data).round()))) + '\n') # confusion_matrix
# n = 0
# model = models.load_model('./Models/model_'+str(n)+'.h5')
# f1.write(np.array2string(np.array(model.evaluate(data, label[:,n]))) + '\n')
# print(label[:,n].shape)
# print(model.predict(data).shape)
# f2.write(np.array2string(np.array(confusion_matrix(label[:,n], model.predict(data).round()))) + '\n') # confusion_matrix
f1.close()
f2.close()
