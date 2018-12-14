import numpy as np
import cv2

def load_batch(ls,od,data_path,batch_size,which_batch):
    image_list=[]
    for i in range(which_batch*batch_size,(which_batch+1)*batch_size):
        image=[]
        image.append(cv2.imread(data_path+ls[od[i]]+'_red.png',0))
        image.append(cv2.imread(data_path+ls[od[i]]+'_green.png',0))
        image.append(cv2.imread(data_path+ls[od[i]]+'_blue.png',0))
        image.append(cv2.imread(data_path+ls[od[i]]+'_yellow.png',0))
        image=np.asarray(image).T
        image_list.append(image)
    image_list=np.asarray(image_list)
    return image_list

def normalize(image_list):
    ma=max(image_list.flatten())
    mi=min(image_list.flatten())
    mean = float((ma+mi)/2.0)
    output = (image_list-mean)/(ma-mean)
    return output
