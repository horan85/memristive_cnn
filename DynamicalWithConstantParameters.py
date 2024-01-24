import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker
import numpy as np
110694773823148758380
5325713563178167160


#testing the dynamical properties of a model with a fixed parameter set
seed=torch.seed()
print(seed)
#this wokrks
#torch.manual_seed(10694773823148758380)

# torch.manual_seed(5325713563178167160)

torch.manual_seed(8651585429011870470)
torch.pi = torch.acos(torch.zeros(1)).item()

A0 =-0.9602
A1 = -0.6280
A2 = -2.1831
A3 = 1.5359
B1 = 1.6686
B2 =1.2363
B3 = -1.1509
C1 =0.5227
C2 = -1.8543
C3 =-0.0450


D0 = -1.2040
D1 = -1.2903
D2 = -1.5158
D3 = -0.2210
E1 =0.6489
E2 = 0.8607
E3 =-0.8444
F1 = 0.2612
F2 = -0.4397
F3 =1.0842
 
 



class MeMCeNNLayer(nn.Module):
    def __init__(self, InDepth=1, OutDepth=1,TimeStep=0.1,IterNum=100):
        super(MeMCeNNLayer, self).__init__()
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.A.weight=nn.Parameter(torch.tensor([[[[-0.1176,  0.0788, -0.1068],
          [-0.7379,  1.1183, -0.8916],
          [ 0.1008, -0.1443,  0.1355]]]],  requires_grad=True))

        self.B= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.B.weight=nn.Parameter(torch.tensor([[[[ 0.0557,  0.1734,  0.2607],
          [-1.5312,  0.6234, -1.2246],
          [-0.2078, -0.2330, -0.0756]]]],  requires_grad=True))

        self.Z= torch.tensor(-1.9875)
        #self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        self.TimeStep=0.3
        self.IterNum=3
        #self.MemStates=torch.zeros([1, OutDepth ,1,2  ]).cuda() 
        #self.InitMemStates=torch.randn([3, OutDepth ,1,2  ]).cuda()
        self.InitMemStates=torch.tensor([0.4841, -0.0494]).reshape(1,OutDepth,1,2)
        
        # 0.4841, -0.0494
        #-0.6606,  0.3078
        #-0.4568, -0.2965
        
        #self.InitMemStates=torch.tensor([-10.0, 10.0]).view(1,1,1,2).cuda() 
       
           
    def LeakyNonLin(self,x,alpha=0.01):
    	y= torch.min(x,1+alpha*(x-1))
    	y= torch.max(y,-1+alpha*(y+1))
    	return y    
        
        
    def MemristiveFunc(self, Vm):
        P1= A0 + A1*Vm +  A2*Vm*Vm +  A3*Vm*Vm*Vm 
        P2= B1*self.MemStates + B2*self.MemStates*self.MemStates+ B3*self.MemStates*self.MemStates*self.MemStates
        P3=C1*Vm*self.MemStates  +  C2*Vm*self.MemStates*self.MemStates  + C3*Vm*Vm*self.MemStates
        self.MemStates= self.MemStates+ self.TimeStep*(P1+P2+P3)
        
        V1=D0 + D1*Vm +  D2*Vm*Vm +  D3*Vm*Vm*Vm 
        V2=E1*self.MemStates+E2*self.MemStates*self.MemStates+ E3*self.MemStates*self.MemStates*self.MemStates
        V3=F1*Vm*self.MemStates  +  F2*Vm*self.MemStates*self.MemStates  + F3*Vm*Vm*self.MemStates
        Out=Vm*(V1 + V2 + V3)
        return Out
           
    def forward(self, x):
        States=torch.zeros((12,1,1,2,self.IterNum))
        self.MemStates=self.InitMemStates
        x=x[:,:,:,1:]
        print(x.shape)
        InputCoupling=self.B(x)
        Zreshaped=self.Z.view(1,InputCoupling.shape[1],1,1).repeat(InputCoupling.shape[0],1,InputCoupling.shape[2],InputCoupling.shape[3])
        InputCoupling=InputCoupling+Zreshaped
        for step in range(self.IterNum):
            Imem=self.MemristiveFunc(x)
            Output=self.LeakyNonLin(x)
            Coupling=self.A(Output) + InputCoupling-Imem
            x=x+self.TimeStep*(-x+Coupling)
            States[:,:,:,:,step]= x.detach()
        return Output, States


class CellNN(nn.Module):
    def __init__(self):
        super(CellNN, self).__init__()
        self.Layer1= MeMCeNNLayer(1,1)
        #self.Layer3= MeMCeNNLayer(4,10)
        

    def forward(self, x):
    	x=self.Layer1(x)
    	#x=self.Layer2(x)
    	#x=self.Layer3(x)
    	return x


def SquaredDiff(NetOut,Labels):
	SquaredDiff=torch.mean(torch.square(NetOut-Labels))
	return  SquaredDiff
	
def SofMaxLoss(NetOut,Labels):
        preds=torch.mean(NetOut,[2,3])
        preds=torch.softmax(preds,-1)
        loss = torch.log(torch.diag(preds[:,Labels]))
        loss =  -torch.mean(loss)
        return loss

     
clf = CellNN()

clf
Memparams=[A0,A1,A2,A3,B1,B2,B3,C1,C2,C3,D0,D1,D2,D3,E1,E2,E3,F1,F2,F3]


data=  torch.tensor([[0,0,0], [0, 0,1], [0,1,0], [0,1,1],[1,0,0], [1,0,1], [1,1,0], [1,1,1],[2,0,0], [2, 0,1], [2,1,0], [2,1,1]]).view(12,1,1,3).float()
response, states=clf(data)
response= torch.sigmoid(response.sum((1,2,3)))
print(response)
states=states.numpy()
print(states.shape)
plt.plot(states[0,0,0,0,:])
plt.show()
