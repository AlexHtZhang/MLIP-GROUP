import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras import models
from keras.layers import Dense
import numpy as np

f = open('demo_result.txt', 'w')
data = np.load("demo_data.npy")
data = np.reshape(data,(1,64,64,8))
real_label = np.load("demo_real_label.npy")
f.write('Expected labels:  ' + np.array2string(real_label) + '\n')
f.write('Predicted labels: [')
counter = 0
for n in range(28):
	model = models.load_model('../assets/multilayer_perceptron_models/model_'+str(n)+'.h5')
	predicted_label = int(model.predict(data)[0][0])
	f.write(np.array2string(np.array(predicted_label)))
	if predicted_label == real_label[n]:
		counter += 1
	if n != 27:
		f.write(' ')
f.write(']\n')
f.write('Accuracy: ' + str(counter * 100 / 28) + '%')
