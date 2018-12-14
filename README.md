Description 
===========
This is project Human Protein Atlas Image Classification developed by team MLIP-GROUP composed of Chenghao, Changtai, Huaqing, Haotian.



Requirements 
============
This project is developed with Python codes v3, using Keras as main deep-learning framework for autoencoder based approach and PyTorch as the main deep-learning framework for the resnet based approach. 

Most of the following packages are already installed on DSMLP, if any is missing, please install. 
Install package 'sklearn, keras, numpy, cv2, matplotlib, pandas, torch' as follow: 

`$ pip install --user sklearn keras numpy opencv-contrib-python matplotlib pandas torch`


Classification Demo  
=================


Re-train
=================
Re-train can be done by downloading the original dataset roughly 17GB. **It can takes you days on a single GTX 1080 Ti.**

In order to retrain, first 


Code organization 
=================
demo.ipynb -- Run a demo of our code (reproduce Figure 3 of our report)

train.ipynb -- Run the training of our model (as described in Section 2) 

code/backprop.py -- Module implementing backprop

code/visu.py -- Module for visualizing our dataset

assets/model.dat Our model trained as described in Section 4





