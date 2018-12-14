import numpy as np 
train=np.load('train.npy')
test=np.load('test.npy')
val=np.load('val.npy')
print(train.shape)
print(test.shape)
print(val.shape)