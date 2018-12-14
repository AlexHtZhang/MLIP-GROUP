import os
import sys
import cnn_ae as ae 


data_path=os.getcwd()+'\\..\\..\\train\\'# Change this to the path to the image folder
image_names_path=os.getcwd()+'\\..\\processed_data\\'
train,test,val=ae.load_filenames(data_path,image_names_path)
train_all=train+test+val#list of total train file name
model=ae.build_model()
batch_size=32
epoch=2
callback=[]
#image_list=ae.load_batch(train_all,data_path,32,0)
#image_list=ae.normalize(image_list)
#callback.append([1,model.fit(image_list,image_list)])
#callback.append([1,model.fit(image_list,image_list)])

for i in range(epoch):
	num_batch=int(len(train_all)/batch_size)-1
	for j in range(num_batch):
		print(str(j)+'/'+str(num_batch))
		image_list=ae.load_batch(train_all,data_path,batch_size,j)
		image_list=ae.normalize(image_list)
		callback.append([i,model.fit(image_list,image_list)])
model.save('auto_encoder.h5')


with open('auto_encoder_loss_log.txt','w') as f:
	for call in callback:
		loss=call[1].history["loss"]
		for s in loss:
			f.write(str(call[0])+' '+str(s)+'\n')
