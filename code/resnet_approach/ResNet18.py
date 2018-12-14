import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# This is our neural networks class that inherits from nn.Module

class ResidualBlock(nn.Module):
    # Here we define our block structure
    def __init__(self, num_in, num_out, stride = 1):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d (num_in, num_out, 3, stride = stride, padding = 1).double(),
            nn.BatchNorm2d (num_out).double(),
            nn.ReLU(inplace=True).double(),
            nn.Conv2d(num_out, num_out, 3, stride = 1, padding = 1).double(),
        )
        self.bn = nn.BatchNorm2d (num_out).double()
        # add residuals
        if num_in != num_out or stride != 1:
            self.res = nn.Sequential(
            nn.Conv2d(num_in,num_out,1,stride = stride).double()
            )
        else:
            self.res = nn.Sequential()
    def forward(self,x):
        out = self.block(x)
        out = out + self.res(x)
        out = self.bn(out)
        out = F.relu(out)
        return out



class ResNet(nn.Module):
    
    # Here we define our network structure
    def __init__( self ):
        super(ResNet , self ). __init__ ()
        self.num_in = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d (4, 64, 7, stride = 2, padding = 3). double(),
            nn.BatchNorm2d(64).double()
        )
        self.layer2 = self.makelayer(ResidualBlock,64,64,1,2)
        self.layer3 = self.makelayer(ResidualBlock,64,128,2,2)
        self.layer4 = self.makelayer(ResidualBlock,128,256,2,2)
        self.layer5 = self.makelayer(ResidualBlock,256,512,2,2)

        self.fc0_1 = nn.Linear (2048, 28).double()
        self.fc0_2 = nn.Linear (28, 2). double ()
        
        self.fc1_1 = nn.Linear (2048, 28).double()
        self.fc1_2 = nn.Linear (28, 2). double ()
        
        self.fc2_1 = nn.Linear (2048, 28).double()
        self.fc2_2 = nn.Linear (28, 2). double ()
        
        self.fc3_1 = nn.Linear (2048, 28).double()
        self.fc3_2 = nn.Linear (28, 2). double ()
        
        self.fc4_1 = nn.Linear (2048, 28).double()
        self.fc4_2 = nn.Linear (28, 2). double ()
        
        self.fc5_1 = nn.Linear (2048, 28).double()
        self.fc5_2 = nn.Linear (28, 2). double ()
        
        self.fc6_1 = nn.Linear (2048, 28).double()
        self.fc6_2 = nn.Linear (28, 2). double ()
        
        self.fc7_1 = nn.Linear (2048, 28).double()
        self.fc7_2 = nn.Linear (28, 2). double ()
        
        self.fc8_1 = nn.Linear (2048, 28).double()
        self.fc8_2 = nn.Linear (28, 2). double ()
        
        self.fc9_1 = nn.Linear (2048, 28).double()
        self.fc9_2 = nn.Linear (28, 2). double ()
        
        self.fc10_1 = nn.Linear (2048, 28).double()
        self.fc10_2 = nn.Linear (28, 2). double ()
        
        self.fc11_1 = nn.Linear (2048, 28).double()
        self.fc11_2 = nn.Linear (28, 2). double ()
        
        self.fc12_1 = nn.Linear (2048, 28).double()
        self.fc12_2 = nn.Linear (28, 2). double ()
        
        self.fc13_1 = nn.Linear (2048, 28).double()
        self.fc13_2 = nn.Linear (28, 2). double ()
        
        self.fc14_1 = nn.Linear (2048, 28).double()
        self.fc14_2 = nn.Linear (28, 2). double ()
        
        self.fc15_1 = nn.Linear (2048, 28).double()
        self.fc15_2 = nn.Linear (28, 2). double ()
        
        self.fc16_1 = nn.Linear (2048, 28).double()
        self.fc16_2 = nn.Linear (28, 2). double ()
        
        self.fc17_1 = nn.Linear (2048, 28).double()
        self.fc17_2 = nn.Linear (28, 2). double ()
        
        self.fc18_1 = nn.Linear (2048, 28).double()
        self.fc18_2 = nn.Linear (28, 2). double ()
        
        self.fc19_1 = nn.Linear (2048, 28).double()
        self.fc19_2 = nn.Linear (28, 2). double ()
        
        self.fc20_1 = nn.Linear (2048, 28).double()
        self.fc20_2 = nn.Linear (28, 2). double ()
        
        self.fc21_1 = nn.Linear (2048, 28).double()
        self.fc21_2 = nn.Linear (28, 2). double ()
        
        self.fc22_1 = nn.Linear (2048, 28).double()
        self.fc22_2 = nn.Linear (28, 2). double ()
        
        self.fc23_1 = nn.Linear (2048, 28).double()
        self.fc23_2 = nn.Linear (28, 2). double ()
        
        self.fc24_1 = nn.Linear (2048, 28).double()
        self.fc24_2 = nn.Linear (28, 2). double ()
        
        self.fc25_1 = nn.Linear (2048, 28).double()
        self.fc25_2 = nn.Linear (28, 2). double ()
        
        self.fc26_1 = nn.Linear (2048, 28).double()
        self.fc26_2 = nn.Linear (28, 2). double ()
        
        self.fc27_1 = nn.Linear (2048, 28).double()
        self.fc27_2 = nn.Linear (28, 2). double ()
        
    def makelayer(self,block,num_in,num_out,stride,k):
        layer = []
        for i in range(k):
            if i == 0:
                layer.append(block(num_in, num_out, stride))
            else:
                layer.append(block(num_out, num_out))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size = 3, stride = 2)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.avg_pool2d(x,8)
        x = x.view(-1,self.num_flat_features(x))
        
        x0 = F.relu(self.fc0_1(x))
        x0 = self.fc0_2(x0)
        
        x1 = F.relu(self.fc1_1(x))
        x1 = self.fc1_2(x1)
        
        x2 = F.relu(self.fc2_1(x))
        x2 = self.fc2_2(x2)
        
        x3 = F.relu(self.fc3_1(x))
        x3 = self.fc3_2(x3)
        
        x4 = F.relu(self.fc4_1(x))
        x4 = self.fc4_2(x4)
        
        x5 = F.relu(self.fc5_1(x))
        x5 = self.fc5_2(x5)
        
        x6 = F.relu(self.fc6_1(x))
        x6 = self.fc6_2(x6)
        
        x7 = F.relu(self.fc7_1(x))
        x7 = self.fc7_2(x7)
        
        x8 = F.relu(self.fc8_1(x))
        x8 = self.fc8_2(x8)
        
        x9 = F.relu(self.fc9_1(x))
        x9 = self.fc9_2(x9)
        
        x10 = F.relu(self.fc10_1(x))
        x10 = self.fc10_2(x10)
        
        x11 = F.relu(self.fc11_1(x))
        x11 = self.fc11_2(x11)
        
        x12 = F.relu(self.fc12_1(x))
        x12 = self.fc12_2(x12)
        
        x13 = F.relu(self.fc13_1(x))
        x13 = self.fc13_2(x13)
        
        x14 = F.relu(self.fc14_1(x))
        x14 = self.fc14_2(x14)
        
        x15 = F.relu(self.fc15_1(x))
        x15 = self.fc15_2(x15)
        
        x16 = F.relu(self.fc16_1(x))
        x16 = self.fc16_2(x16)
        
        x17 = F.relu(self.fc17_1(x))
        x17 = self.fc17_2(x17)
        
        x18 = F.relu(self.fc18_1(x))
        x18 = self.fc18_2(x18)
        
        x19 = F.relu(self.fc19_1(x))
        x19 = self.fc19_2(x19)
        
        x20 = F.relu(self.fc20_1(x))
        x20 = self.fc20_2(x20)
        
        x21 = F.relu(self.fc21_1(x))
        x21 = self.fc21_2(x21)
        
        x22 = F.relu(self.fc22_1(x))
        x22 = self.fc22_2(x22)
        
        x23 = F.relu(self.fc23_1(x))
        x23 = self.fc23_2(x23)
        
        x24 = F.relu(self.fc24_1(x))
        x24 = self.fc24_2(x24)
        
        x25 = F.relu(self.fc25_1(x))
        x25 = self.fc25_2(x25)
        
        x26 = F.relu(self.fc26_1(x))
        x26 = self.fc26_2(x26)
        
        x27 = F.relu(self.fc27_1(x))
        x27 = self.fc27_2(x27)
        
        return x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27
        
    # Determine the number of features in a batch of tensors
    def num_flat_features (self , x):
        size = x. size ()[1:]
        return np. prod ( size )
