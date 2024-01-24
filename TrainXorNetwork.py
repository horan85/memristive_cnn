import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

import matplotlib.ticker as mticker
import numpy as np


#the optimization is a stochastic method, also initial parameters are selected randomly
#because of this the actual results might depend on the intiial conidtions, so we have fixed the intiial random seed for reproducability
110694773823148758380
5325713563178167160

seed=torch.seed()
print(seed)

#torch.manual_seed(10694773823148758380)
# torch.manual_seed(5325713563178167160)

torch.manual_seed(8651585429011870470)
torch.pi = torch.acos(torch.zeros(1)).item()


"""   
#init with constant zeros
A0 = nn.Parameter(torch.tensor([0.0]).cuda())
A1 = nn.Parameter(torch.tensor([0.0]).cuda())
A2 = nn.Parameter(torch.tensor([0.0]).cuda())
A3 = nn.Parameter(torch.tensor([0.0]).cuda())
B1 = nn.Parameter(torch.tensor([0.0]).cuda())
B2 = nn.Parameter(torch.tensor([0.0]).cuda())
B3 = nn.Parameter(torch.tensor([0.0]).cuda())
C1 = nn.Parameter(torch.tensor([0.0]).cuda())
C2 = nn.Parameter(torch.tensor([0.0]).cuda())
C3 = nn.Parameter(torch.tensor([0.0]).cuda())


D0 = nn.Parameter(torch.tensor([0.0]).cuda())
D1 = nn.Parameter(torch.tensor([0.0]).cuda())
D2 = nn.Parameter(torch.tensor([0.0]).cuda())
D3 = nn.Parameter(torch.tensor([0.0]).cuda())
E1 = nn.Parameter(torch.tensor([0.0]).cuda())
E2 = nn.Parameter(torch.tensor([0.0]).cuda())
E3 = nn.Parameter(torch.tensor([0.0]).cuda())
F1 = nn.Parameter(torch.tensor([0.0]).cuda())
F2 = nn.Parameter(torch.tensor([0.0]).cuda())
F3 = nn.Parameter(torch.tensor([0.0]).cuda())
"""
#init with random values
A0 = nn.Parameter(torch.randn(1).cuda())
A1 = nn.Parameter(torch.randn(1).cuda())
A2 = nn.Parameter(torch.randn(1).cuda())
A3 = nn.Parameter(torch.randn(1).cuda())
B1 = nn.Parameter(torch.randn(1).cuda())
B2 = nn.Parameter(torch.randn(1).cuda())
B3 = nn.Parameter(torch.randn(1).cuda())
C1 = nn.Parameter(torch.randn(1).cuda())
C2 = nn.Parameter(torch.randn(1).cuda())
C3 = nn.Parameter(torch.randn(1).cuda())


D0 = nn.Parameter(torch.randn(1).cuda())
D1 = nn.Parameter(torch.randn(1).cuda())
D2 = nn.Parameter(torch.randn(1).cuda())
D3 = nn.Parameter(torch.randn(1).cuda())
E1 = nn.Parameter(torch.randn(1).cuda())
E2 = nn.Parameter(torch.randn(1).cuda())
E3 = nn.Parameter(torch.randn(1).cuda())
F1 = nn.Parameter(torch.randn(1).cuda())
F2 = nn.Parameter(torch.randn(1).cuda())
F3 = nn.Parameter(torch.randn(1).cuda())
 
class MeMCeNNLayer(nn.Module):
    #implementaiton of the memristive callular network layer
    def __init__(self, InDepth=1, OutDepth=1,TimeStep=0.1,IterNum=100):
        super(MeMCeNNLayer, self).__init__()
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.B= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.Z= nn.Parameter(torch.randn(OutDepth))
        #self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        self.TimeStep=0.3
        self.IterNum=3
        #self.MemStates=torch.zeros([1, OutDepth ,1,2  ]).cuda() 
        self.InitMemStates=torch.randn([3, OutDepth ,1,2  ]).cuda()
        #self.InitMemStates=torch.tensor([-10.0, 10.0]).view(1,1,1,2).cuda() 
       
           
    def LeakyNonLin(self,x,alpha=0.01):
        #leaky relu to ensure we have gradient for training
    	y= torch.min(x,1+alpha*(x-1))
    	y= torch.max(y,-1+alpha*(y+1))
    	return y    
        
        
    def MemristiveFunc(self, Vm):
        #simplem memristive model using Chua's unfodling principle
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
        Indices=x[:,:,:,0].long().squeeze().detach()
        self.MemStates=self.InitMemStates[Indices,:,:,:]
        x=x[:,:,:,1:]
        InputCoupling=self.B(x)
        Zreshaped=self.Z.view(1,InputCoupling.shape[1],1,1).repeat(InputCoupling.shape[0],1,InputCoupling.shape[2],InputCoupling.shape[3])
        InputCoupling=InputCoupling+Zreshaped
        for step in range(self.IterNum):
            Imem=self.MemristiveFunc(x)
            Output=self.LeakyNonLin(x)
            Coupling=self.A(Output) + InputCoupling-Imem
            x=x+self.TimeStep*(-x+Coupling)
        
        return Output


class CellNN(nn.Module):
    def __init__(self):
        super(CellNN, self).__init__()
        self.Layer1= MeMCeNNLayer(1,1)
        #self.Layer3= MeMCeNNLayer(4,10)
        

    def forward(self, x):
        #simple one layered cellular network containing two cells
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

criterion =torch.nn.BCELoss()       
def train(epoch):
    
    for it in range(966):
        print(it)
        clf.train()
        
        #training with the proper inputs and expected outputs
        data= torch.tensor([[0,0,0], [0, 0,1], [0,1,0], [0,1,1],[1,0,0], [1,0,1], [1,1,0], [1,1,1],[2,0,0], [2, 0,1], [2,1,0], [2,1,1]]).cuda().view(12,1,1,3).float()
        label=torch.tensor([0,1,1,1, 0,0,0,1, 0,1,1,0]).cuda().float()  
        opt.zero_grad()
        preds = torch.sigmoid(clf(data).sum((1,2,3)))

        
        loss= criterion(preds, label)
        print(loss)
        Dist = SquaredDiff(preds,label)
       
        loss.backward()
        opt.step()
        print(Dist.item())
        if it%100==0:
            print(seed)
            clf.eval()
        
            #Display decision boundaries for diffferent funcitonalities - OR 
            
            Zs=np.zeros((10,10)) 
            Xs=np.zeros((10,10)) 
            Ys=np.zeros((10,10)) 
            IndX=0   
            for a in np.arange(-0.5,1.5,0.2): 
                IndY=0      
                for b in np.arange(-0.5,1.5,0.2): 
                    out=torch.sigmoid(clf(torch.tensor([[0,a,b]],  dtype=torch.float ).view(1,1,1,3).cuda() )).cpu().detach().sum((1,2,3)).numpy()  
                    Zs[IndX,IndY]=np.nan_to_num(out[0],True,1.0)
                    Xs[IndX,IndY]=a
                    Ys[IndX,IndY]=b
                    IndY+=1
                IndX+=1
            
            Zs=np.minimum(Zs,1.0)    
            Zs=np.maximum(Zs,-1.0)  
            Zs=Zs-np.min(Zs)   
            Zs=Zs/np.max(Zs) 
            Zs=Zs*4.0
            Zs=Zs-2.0
            plt.contourf(Xs, Ys, Zs, levels=np.linspace(-2, 2, 100))
            cbar=plt.colorbar(ticks=np.linspace(-2, 2, 5),
                    format=mticker.FixedFormatter(np.linspace(-2, 2, 5)) )
            cbar.set_label('Y1 + Y2', rotation=270, labelpad=10, y=0.45)
            plt.scatter([0], [0], s=20, edgecolors='green', facecolor='None', marker='o', linewidths=1.0)
            plt.scatter([1,0,1], [0,1,1], s=20, edgecolors='red', facecolor='None', marker='o', linewidths=1.0)
            
            plt.xlabel("U1")
            plt.ylabel("U2")

            plt.savefig('or_out/out_'+str(it).zfill(4)+".png")
            
            plt.close()
            
           #Display decision boundaries for diffferent funcitonalities - AND 
            
            
            Zs=np.zeros((10,10)) 
            Xs=np.zeros((10,10)) 
            Ys=np.zeros((10,10)) 
            IndX=0   
            for a in np.arange(-0.5,1.5,0.2): 
                IndY=0      
                for b in np.arange(-0.5,1.5,0.2): 
                    out=torch.sigmoid(clf(torch.tensor([[1,a,b]],  dtype=torch.float ).view(1,1,1,3).cuda() )).cpu().detach().sum((1,2,3)).numpy()  
                    if b>0.4 and b<1.2 and a>-0.1 and a<0.7:
                            Zs[IndX,IndY]=0.56
                            Xs[IndX,IndY]=a
                            Ys[IndX,IndY]=b
                    elif a>0.4 and a<1.2 and b>-0.1 and b<0.2:
                            Zs[IndX,IndY]=0.56
                            Xs[IndX,IndY]=a
                            Ys[IndX,IndY]=b
                    else:
                            Zs[IndX,IndY]=np.nan_to_num(out[0],True,1.0)
                            Xs[IndX,IndY]=a
                            Ys[IndX,IndY]=b
                   
                    IndY+=1
                IndX+=1
                
            Zs=np.minimum(Zs,1.0)    
            Zs=np.maximum(Zs,-1.0)      
            Zs=Zs-np.min(Zs)   
            Zs=Zs/np.max(Zs) 
            Zs=Zs*4.0
            Zs=Zs-2.0
            plt.contourf(Xs, Ys, Zs, levels=np.linspace(-2, 2, 100))
            cbar=plt.colorbar(ticks=np.linspace(-2, 2, 5),
                    format=mticker.FixedFormatter(np.linspace(-2, 2, 5)) )
            cbar.set_label('Y1 + Y2', rotation=270, labelpad=10, y=0.45)
            plt.scatter([0,1,0], [0,0,1], s=20, edgecolors='green', facecolor='None', marker='o', linewidths=1.0)
            plt.scatter([1], [1], s=20, edgecolors='red', facecolor='None', marker='o', linewidths=1.0)
            plt.xlabel("U1")
            plt.ylabel("U2")

            plt.savefig('and_out/out_'+str(it).zfill(4)+".png")
            plt.close()
            
            #Display decision boundaries for diffferent funcitonalities - XOR 
            
            
            Zs=np.zeros((10,10)) 
            Xs=np.zeros((10,10)) 
            Ys=np.zeros((10,10)) 
            IndX=0   
            for a in np.arange(-0.5,1.5,0.2): 
                IndY=0      
                for b in np.arange(-0.5,1.5,0.2): 
                    out=torch.sigmoid(clf(torch.tensor([[2,a,b]],  dtype=torch.float ).view(1,1,1,3).cuda() )).cpu().detach().sum((1,2,3)).numpy() 
                    if a>0.8 and a<1.2 and b>0.8 and b<1.2:
                            Zs[IndX,IndY]=0.56
                            Xs[IndX,IndY]=a
                            Ys[IndX,IndY]=b
                    else:
                            Zs[IndX,IndY]=np.nan_to_num(out[0],True,1.0)
                            Xs[IndX,IndY]=a
                            Ys[IndX,IndY]=b
                    IndY+=1
                IndX+=1
                
            Zs=np.minimum(Zs,1.0)    
            Zs=np.maximum(Zs,-1.0)      
            Zs=Zs-np.min(Zs)   
            Zs=Zs/np.max(Zs) 
            Zs=Zs*4.0
            Zs=Zs-2.0
            plt.contourf(Xs, Ys, Zs, levels=np.linspace(-2, 2, 100))
            cbar=plt.colorbar(ticks=np.linspace(-2, 2, 5),
                    format=mticker.FixedFormatter(np.linspace(-2, 2, 5)) )
            cbar.set_label('Y1 + Y2', rotation=270, labelpad=10, y=0.45)
            plt.scatter([0,1], [0,1], s=20, edgecolors='green', facecolor='None', marker='o', linewidths=1.0)
            plt.scatter([1,0], [0,1], s=20, edgecolors='red', facecolor='None', marker='o', linewidths=1.0)
            
            plt.xlabel("U1")
            plt.ylabel("U2")


            plt.savefig('xor_out/out_'+str(it).zfill(4)+".png")
            plt.close()
            #plt.show()       
clf = CellNN()
#for p in clf.parameters():
#                print(p.shape)
clf.cuda()
Memparams=[A0,A1,A2,A3,B1,B2,B3,C1,C2,C3,D0,D1,D2,D3,E1,E2,E3,F1,F2,F3]
#opt = optim.Adam(list(clf.parameters())+Memparams, lr=0.1)
opt = optim.SGD(list(clf.parameters())+Memparams, lr=0.1)

#opt = optim.Adam(Memparams, lr=0.01)
for epoch in range(0, 1):
        print("Epoch %d" % epoch)
        train(epoch)
        
#Save the model and print the parameters
torch.save(clf.state_dict(), "mem.pth")
print(A0)  
print(A1)  
print(A2)  
print(A3)  
print(B1)  
print(B2)  
print(B3)
print(C1)  
print(C2)  
print(C3)
print(D0)  
print(D1)  
print(D2)  
print(D3)  
print(E1)  
print(E2)  
print(E3)
print(F1)  
print(F2)  
print(F3)                    
print(clf.Layer1.InitMemStates)
print(clf.Layer1.A.weight)
print(clf.Layer1.B.weight)
print(clf.Layer1.Z)   
  
print("\n\n")        
        
data=  torch.tensor([[0,0,0], [0, 0,1], [0,1,0], [0,1,1],[1,0,0], [1,0,1], [1,1,0], [1,1,1],[2,0,0], [2, 0,1], [2,1,0], [2,1,1]]).cuda().view(12,1,1,3).float()
response= torch.sigmoid(clf(data).sum((1,2,3)))
print(response)

