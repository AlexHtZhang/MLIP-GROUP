import torch
import pandas as pd
import numpy as np
import os
import torch.autograd as ag
import torch.nn as nn
import torch.nn.functional as F

import prepare
import ResNet18

path = os.getcwd()# we can change the path to fit our need
data_path = os.path.join(path, 'human-protein/train/')# this path is temporary

#order of training set
df = pd.read_csv(path+'/data/train_idx.txt', sep = '\t', header = None)
df.columns = ['order']
order = list(df['order'])

# labels of one hot
label = np.load(path +'/data/train_one_hot.npy')

# id of pictures
trainset = pd.read_csv(path + '/data/train.csv')
ls = trainset['Id']
num = trainset['Target']

#hyper-parameter
N = label.shape[0] # Training set size
B = 28             # Minibacth size
NB = int(N/B)-1        # Number of minibatches
T = 2               # Number of epochs
criterion = nn.CrossEntropyLoss ()

# training preparation
if torch.cuda.is_available ():
    net = ResNet18.ResNet().cuda()
    ltrain = ag.Variable(torch.from_numpy(label).cuda(),requires_grad = False)
optimizer = torch.optim .SGD(net. parameters(),lr= 0.001 ,momentum = 0.9)

# start training
for epoch in range(T):
    running_loss = 0.0

    for k in range(NB):

        idxsmp = k*B # indices of samples for k-th minibatch

        xt = prepare.load_batch(ls,order,data_path, B, k)
        xt = prepare.normalize(xt)
        xtrain = np.moveaxis(xt,[1,2],[2,3])

        inputs = ag.Variable(torch.from_numpy(xtrain).cuda(),requires_grad = True)

        labels0 = ltrain [ idxsmp:idxsmp+B,0 ]
        labels1 = ltrain [ idxsmp:idxsmp+B,1 ]
        labels2 = ltrain [ idxsmp:idxsmp+B,2 ]
        labels3 = ltrain [ idxsmp:idxsmp+B,3 ]
        labels4 = ltrain [ idxsmp:idxsmp+B,4 ]
        labels5 = ltrain [ idxsmp:idxsmp+B,5 ]
        labels6 = ltrain [ idxsmp:idxsmp+B,6 ]
        labels7 = ltrain [ idxsmp:idxsmp+B,7 ]
        labels8 = ltrain [ idxsmp:idxsmp+B,8 ]
        labels9 = ltrain [ idxsmp:idxsmp+B,9 ]
        labels10 = ltrain [ idxsmp:idxsmp+B,10 ]
        labels11 = ltrain [ idxsmp:idxsmp+B,11 ]
        labels12 = ltrain [ idxsmp:idxsmp+B,12 ]
        labels13 = ltrain [ idxsmp:idxsmp+B,13 ]
        labels14 = ltrain [ idxsmp:idxsmp+B,14 ]
        labels15 = ltrain [ idxsmp:idxsmp+B,15 ]
        labels16 = ltrain [ idxsmp:idxsmp+B,16 ]
        labels17 = ltrain [ idxsmp:idxsmp+B,17 ]
        labels18 = ltrain [ idxsmp:idxsmp+B,18 ]
        labels19 = ltrain [ idxsmp:idxsmp+B,19 ]
        labels20 = ltrain [ idxsmp:idxsmp+B,20 ]
        labels21 = ltrain [ idxsmp:idxsmp+B,21 ]
        labels22 = ltrain [ idxsmp:idxsmp+B,22 ]
        labels23 = ltrain [ idxsmp:idxsmp+B,23 ]
        labels24 = ltrain [ idxsmp:idxsmp+B,24 ]
        labels25 = ltrain [ idxsmp:idxsmp+B,25 ]
        labels26 = ltrain [ idxsmp:idxsmp+B,26 ]
        labels27 = ltrain [ idxsmp:idxsmp+B,27 ]

        # Initialize the gradients to zero
        optimizer.zero_grad ()
        # Forward propagation
        x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27 = net ( inputs )

        # Error evaluation
        loss0 = criterion( x0 , labels0 )
        loss1 = criterion( x1 , labels1 )
        loss2 = criterion( x2 , labels2 )
        loss3 = criterion( x3 , labels3 )
        loss4 = criterion( x4 , labels4 )
        loss5 = criterion( x5 , labels5 )
        loss6 = criterion( x6 , labels6 )
        loss7 = criterion( x7 , labels7 )
        loss8 = criterion( x8 , labels8 )
        loss9 = criterion( x9 , labels9 )
        loss10 = criterion( x10 , labels10 )
        loss11 = criterion( x11 , labels11 )
        loss12 = criterion( x12 , labels12 )
        loss13 = criterion( x13 , labels13 )
        loss14 = criterion( x14 , labels14 )
        loss15 = criterion( x15 , labels15 )
        loss16 = criterion( x16 , labels16 )
        loss17 = criterion( x17 , labels17 )
        loss18 = criterion( x18 , labels18 )
        loss19 = criterion( x19 , labels19 )
        loss20 = criterion( x20 , labels20 )
        loss21 = criterion( x21 , labels21 )
        loss22 = criterion( x22 , labels22 )
        loss23 = criterion( x23 , labels23 )
        loss24 = criterion( x24 , labels24 )
        loss25 = criterion( x25 , labels25 )
        loss26 = criterion( x26 , labels26 )
        loss27 = criterion( x27 , labels27 )
        loss_01 = loss0+loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10
        loss_02 = loss11+loss12+loss13+loss14+loss15+loss16+loss17+loss18+loss19+loss20
        loss_03 = loss21+loss22+loss23+loss24+loss25+loss26+loss27
        loss = loss_01+loss_02+loss_03

        # Back propagation
        loss.backward() #retain_graph=True)
        # Parameter update
        optimizer.step ()
        # Print averaged loss per minibatch every 100 mini - batches
        running_loss += loss.cpu().data[0]

        if k % 100 == 99:
            print ('[%d, %5d] loss : %.3f' %( epoch + 1, k + 1, running_loss/100 ))
            running_loss = 0.0
            
    torch.save(net, 'ResNet18_28outputs_epoch%d.pkl'%(epoch+1))
    
print ('Finished Training ')
