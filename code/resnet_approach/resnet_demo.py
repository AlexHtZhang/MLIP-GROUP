# coding: utf-8
import torch
import pandas as pd
import numpy as np
import cv2
import os
import torch.autograd as ag

import prepare
from ResNet18 import ResNet
from ResNet18 import ResidualBlock


def resnet_demo():
    path = os.getcwd()# we can change the path to fit our need

    print('loading first sample in the test set')
    data_path = '../..' + '/human-protein/train/' # this path is temporary

    df = pd.read_csv('../processed_data/test_idx.txt', sep = '\t', header = None)
    df.columns = ['order']
    order = list(df['order'])

    # labels of one hot
    label = np.load('../processed_data/test_one_hot.npy')

    # id of pictures
    testset = pd.read_csv('../processed_data/train.csv')
    ls = testset['Id']
    num = testset['Target']
    print('loading resnet18 weights')
    net = torch.load('ResNet18_28outputs_epoch2.pkl')#upload the net

    N = label.shape[0] # Training set size
    B = 1             # Minibacth size
    NB = int(N/B)-1 
    out0=out1=out2=out3=out4=out5=out6=out7=out8=out9=out10=out11 = 0
    out12=out13=out14=out15=out16=out17=out18=out19=out20=out21 = 0
    out22=out23=out24=out25=out26=out27 = 0

    def load_batch(name,data_path,batch_size,which_batch):
        image_list=[]
        image=[]
        image.append(cv2.imread(data_path+name+'_red.png',0))
        image.append(cv2.imread(data_path+name+'_green.png',0))
        image.append(cv2.imread(data_path+name+'_blue.png',0))
        image.append(cv2.imread(data_path+name+'_yellow.png',0))
        image=np.asarray(image).T
        image_list.append(image)
        image_list=np.asarray(image_list)
        return image_list

    k = i = 0
    xtest = load_batch('00070df0-bbc3-11e8-b2bc-ac1f6b6435d0',data_path,B,k)
    xtest = prepare.normalize(xtest)
    xtest = np.moveaxis(xtest,[1,2],[2,3])
    xtest = ag.Variable( torch.from_numpy ( xtest ), requires_grad = False )
    #get results
    y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20,y21,y22,y23,y24,y25,y26,y27  = net(xtest.cuda())

    out0 = out0 + np.sum(label[i:i+B,0] == y0.cpu().data.numpy().T.argmax(axis =0))
    out1 = out1 + np.sum(label[i:i+B,1] == y1.cpu().data.numpy().T.argmax(axis =0))
    out2 = out2 + np.sum(label[i:i+B,2] == y2.cpu().data.numpy().T.argmax(axis =0))
    out3 = out3 + np.sum(label[i:i+B,3] == y3.cpu().data.numpy().T.argmax(axis =0))
    out4 = out4 + np.sum(label[i:i+B,4] == y4.cpu().data.numpy().T.argmax(axis =0))
    out5 = out5 + np.sum(label[i:i+B,5] == y5.cpu().data.numpy().T.argmax(axis =0))
    out6 = out6 + np.sum(label[i:i+B,6] == y6.cpu().data.numpy().T.argmax(axis =0))
    out7 = out7 + np.sum(label[i:i+B,7] == y7.cpu().data.numpy().T.argmax(axis =0))
    out8 = out8 + np.sum(label[i:i+B,8] == y8.cpu().data.numpy().T.argmax(axis =0))
    out9 = out9 + np.sum(label[i:i+B,9] == y9.cpu().data.numpy().T.argmax(axis =0))
    out10 = out10 + np.sum(label[i:i+B,10] == y10.cpu().data.numpy().T.argmax(axis =0))
    out11 = out11 + np.sum(label[i:i+B,11] == y11.cpu().data.numpy().T.argmax(axis =0))
    out12 = out12 + np.sum(label[i:i+B,12] == y12.cpu().data.numpy().T.argmax(axis =0))
    out13 = out13 + np.sum(label[i:i+B,13] == y13.cpu().data.numpy().T.argmax(axis =0))
    out14 = out14 + np.sum(label[i:i+B,14] == y14.cpu().data.numpy().T.argmax(axis =0))
    out15 = out15 + np.sum(label[i:i+B,15] == y15.cpu().data.numpy().T.argmax(axis =0))
    out16 = out16 + np.sum(label[i:i+B,16] == y16.cpu().data.numpy().T.argmax(axis =0))
    out17 = out17 + np.sum(label[i:i+B,17] == y17.cpu().data.numpy().T.argmax(axis =0))
    out18 = out18 + np.sum(label[i:i+B,18] == y18.cpu().data.numpy().T.argmax(axis =0))
    out19 = out19 + np.sum(label[i:i+B,19] == y19.cpu().data.numpy().T.argmax(axis =0))
    out20 = out20 + np.sum(label[i:i+B,20] == y20.cpu().data.numpy().T.argmax(axis =0))
    out21 = out21 + np.sum(label[i:i+B,21] == y21.cpu().data.numpy().T.argmax(axis =0))
    out22 = out22 + np.sum(label[i:i+B,22] == y22.cpu().data.numpy().T.argmax(axis =0))
    out23 = out23 + np.sum(label[i:i+B,23] == y23.cpu().data.numpy().T.argmax(axis =0))
    out24 = out24 + np.sum(label[i:i+B,24] == y24.cpu().data.numpy().T.argmax(axis =0))
    out25 = out25 + np.sum(label[i:i+B,25] == y25.cpu().data.numpy().T.argmax(axis =0))
    out26 = out26 + np.sum(label[i:i+B,26] == y26.cpu().data.numpy().T.argmax(axis =0))
    out27 = out27 + np.sum(label[i:i+B,27] == y27.cpu().data.numpy().T.argmax(axis =0))


    res=[out0,out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11,out12,out13,out14,out15,out16,out17,out18,out19,out20,out21,out22,out23,out24,out25,out26,out27]

    average_accuracy = sum(res) * 1./28
    def flip(prediction):
        if prediction == 0:
            return 1
        else:
            return 0
    res = [ flip(prediction) for prediction in res]
    print('done prediction:')
    print(res)#accuracy of every class
    print('accuracy: ')
    print(average_accuracy)

    return 