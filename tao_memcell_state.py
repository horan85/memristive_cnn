import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import cv2

torch.pi = torch.acos(torch.zeros(1)).item()


batch_size=32

# download and transform train dataset
train_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=True, transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
batch_size=batch_size, shuffle=True)

# download and transform test dataset
test_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/horan/Data/mnist_data', download=True, train=False,transform=transforms.Compose([
transforms.ToTensor(), # first, convert image to PyTorch tensor
transforms.Normalize((0.1307,), (0.3081,)) ])), 
 batch_size=batch_size, shuffle=True)

class MeMCeNNLayer(nn.Module):
    def __init__(self, InDepth=1, OutDepth=1,TimeStep=0.1,IterNum=100):
        super(MeMCeNNLayer, self).__init__()
        self.OutDepth=OutDepth
        self.rescalex= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.rescalev= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1)
        self.B= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1)
        self.Z= nn.Parameter(torch.randn(OutDepth))
        #self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        
        
        self.TimeStep=0.01
        self.IterNum=10
        
        
        # on switching v>0
        self.Bb=1e-4
        self.sigma_on=.45
        self.y_on=.06
        self.sigma_p=4e-5
        # off switching v<0
        self.Aa=1e-10
        self.sigma_off=0.013
        self.y_off=1.0
        self.beta=500
        #parameters of memductance 
        self.Gm=0.025
        self.a=7.2e-6
        self.b=4.7
        #input parameters
        self.v0=1.0
    
    
    def mem(self,x1,x2):

      #x1=y state of TaO memristor
      #x2 time
      torch.exp(self.b*torch.sqrt(torch.abs(x2)))*(1-x1)
      out=self.Gm*x1+self.a*torch.exp(self.b*torch.sqrt(torch.abs(x2)))*(1-x1)
      return out
      
    def stepfunc(self,x):
      return (1+torch.sign(x))/2


  
    def NonLin(self,x,alpha=0.01):
    	y= torch.min(x,1+alpha*(x-1))
    	y= torch.max(y,-1+alpha*(y+1))
    	return y
           
    def forward(self, x,v):
        
        Zreshaped=self.Z.view(1,self.OutDepth,1,1).repeat(x.shape[0],1,x.shape[2],x.shape[3])
        x=self.rescalex(x)
        v=self.rescalev(v)
        
        for step in range(self.IterNum):

                i=self.NonLin(v)
     
                InputCoupling=self.B(i) 
                OutputCoupling=self.A(self.NonLin(i)) 
                Coupling=InputCoupling+OutputCoupling+Zreshaped
                
                
                vm=v
                im= self.mem(x,vm)*vm
                p=im*vm;
              
                vdot=1
                
                xdot= self.Aa*torch.sinh(torch.tensor(vm/self.sigma_off))*torch.exp(-torch.square(self.y_off/x))*torch.exp(1/(1+self.beta*p))*self.stepfunc(-vm)+self.Bb*torch.sinh(vm/self.sigma_on)*torch.exp(-torch.square(x/self.y_on))*torch.exp(p/self.sigma_p)*self.stepfunc(vm)  +Coupling
                
                x=x+self.TimeStep*(xdot)
                v=v+self.TimeStep*(vdot)
            
        return x, v

class CeNNLayer(nn.Module):
    def __init__(self, InDepth=1, OutDepth=1,TimeStep=0.1,IterNum=100):
        super(CeNNLayer, self).__init__()
        self.rescale= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1)
        self.B= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.Z= nn.Parameter(torch.randn(OutDepth))
        #self.Z =self.Zsingle.view(1,OutDepth,1,1).repeat(16,1,28,28)
        self.TimeStep=0.01
        self.IterNum=10
        
    def NonLin(self,x,alpha=0.01):
    	y= torch.min(x,1+alpha*(x-1))
    	y= torch.max(y,-1+alpha*(y+1))
    	return y
           
    def forward(self, x):
        InputCoupling=self.B(x)
        Zreshaped=self.Z.view(1,InputCoupling.shape[1],1,1).repeat(InputCoupling.shape[0],1,InputCoupling.shape[2],InputCoupling.shape[3])
        InputCoupling=InputCoupling+Zreshaped
        x=self.rescale(x)
        for step in range(self.IterNum):
            Coupling=self.A(self.NonLin(x)) + InputCoupling
            x=x+self.TimeStep*(-x+Coupling)
        return self.NonLin(x)


class CellNN(nn.Module):
    def __init__(self):
        super(CellNN, self).__init__()
        self.Layer1= MeMCeNNLayer(1,16)
        self.Layer2= MeMCeNNLayer(16,32)
        self.Layer3= MeMCeNNLayer(32,10)
        

    def forward(self, x,v):
    	x,v=self.Layer1(x,v)
    	x,v=self.Layer2(x,v)
    	x,v=self.Layer3(x,v)
    	return v


def SquaredDiff(NetOut,Labels):
	SquaredDiff=torch.mean(torch.square(NetOut-Labels))
	return  SquaredDiff
	
def SofMaxLoss(NetOut,Labels):
        preds=torch.mean(NetOut,[2,3])
        preds=torch.softmax(preds,-1)
        loss = torch.log(torch.diag(preds[:,Labels]))
        loss =  -torch.mean(loss)
        return loss
        
def train(epoch):
    for batch_id, (data, label) in enumerate(train_loader):
        clf.train()
        data=data.cuda()
        label=label.cuda()
        opt.zero_grad()
        preds = clf(data,torch.ones_like(data).cuda())
        
        #one_hot = torch.zeros(preds.shape[0], preds.shape[1]).cuda()
        #one_hot[torch.arange(preds.shape[0]), label] = 1
        #ImgLabels=one_hot.view(preds.shape[0], preds.shape[1],1,1).repeat(1,1,preds.shape[2],preds.shape[3])
        #ImgLabels=(2*ImgLabels-1.0)
  
        #loss = SquaredDiff(preds,ImgLabels)
        loss = SofMaxLoss(preds,label)
        
        loss.backward()
        opt.step()
        #probs=torch.softmax(preds,-1)
        predind = torch.sum(preds, [2,3])
        predind = predind.data.max(1)[1] 
        acc = predind.eq(label.data).cpu().float().mean() 

        if batch_id % 100 == 0:
            
            print("Train Loss: "+str(loss.item())+" Acc: "+str(acc.item()))
            #run independent test
            clf.eval() # set model in inference mode (need this because of dropout)
            test_loss = 0
            correct = 0
            SampleNum=0
            for data, target in test_loader: 
                if data.shape[0]==batch_size:
                        data=data.cuda()
                        label=target.cuda()  
                        with torch.no_grad():    
                           output = clf(data,torch.ones_like(data).cuda())
                           #probs=torch.softmax(output,-1)
                           pred = torch.sum(output, [2,3]).data.max(1)[1] 
                           correct += pred.eq(label.data).cpu().sum()
                        SampleNum+=data.shape[0]
            accuracy =  correct.item() / SampleNum
            print("Test Acc: "+str(accuracy))
            
clf = CellNN()
#for p in clf.parameters():
#                print(p.shape)
clf.cuda()
opt = optim.Adam(clf.parameters(), lr=0.001)
for epoch in range(0, 10):
        print("Epoch %d" % epoch)
        train(epoch)
  
