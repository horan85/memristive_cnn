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

class MeMCeNNLayer(nn.Module):
    def __init__(self,  InDepth=1, OutDepth=1,TimeStep=0.01,IterNum=100, nonlin='mem', InputShape=[1,1,1]):
        super(MeMCeNNLayer, self).__init__()
        
        self.rescale= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1)
        self.A= nn.Conv2d(OutDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.B= nn.Conv2d(InDepth, OutDepth, kernel_size=3, padding=1, bias=False)
        self.Z= nn.Parameter(torch.randn(OutDepth))
        
        self.TimeStep=0.1
        self.IterNum=1
        #self.MemStates=torch.zeros([1, OutDepth ,1,2  ]).cuda() 
        #self.InitMemStates=nn.Parameter( torch.rand([1, OutDepth ,1,2  ] )*0.1 )
        #self.InitMemStates=   nn.Parameter( torch.randn([1, OutDepth ,1,2  ] ) *0.1 )
        self.InitMemStates=(torch.randn([1, InputShape[0] , InputShape[1], InputShape[2]  ])*0.1).cuda() 
        #self.InitMemStates=nn.Parameter( torch.ones([1, OutDepth ,1,2  ] )*0.0001 )
        #self.InitMemStates= torch.ones([1, OutDepth ,1,2  ] )*0.1 
        
        #self.InitMemStates=torch.tensor([-10.0, 10.0]).view(1,1,1,2).cuda() 
       
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
        """
        self.Vt=0.95
        self.alpha=1e5
        self.beta=1e6
        self.p=1
        self.xon=2000
        self.xoff=10000
        
        self.Cx =1.0 #10 micro
        self.Ry=1.0
        self.glin=1.0  #1 mili
        """
        
    def LeakyNonLin(self,x,alpha=0.001):
    	NonlinPoint=0.1
    	y= torch.min(x,NonlinPoint+alpha*(x-NonlinPoint))
    	y= torch.max(y,-NonlinPoint+alpha*(y+NonlinPoint))
    	return y
    
    def MemFunc(self,x,v):
      i=self.glin* self.NonLin(v)
      k=-self.beta *v +(self.beta-self.alpha)/2* ( abs(v+self.Vt)-abs(v-self.Vt))
      fpx= 1- (( x -self.xon)/ (self.xoff-self.xon)-1)
      fmx= 1- (( x -self.xon)/ (self.xoff-self.xon) )
         
      vdot=(-( torch.clamp(1/(x),-0.1,0.1) )*v +Coupling)/self.Cx
                
      xdot= k*( (v>0)* fpx + (v<0)*fmx  ) 
      vdot=torch.clamp(vdot,-10,10)
      xdot=torch.clamp(xdot,-10,10)
      x=x+self.TimeStep*(xdot)
      v=v+self.TimeStep*(vdot)          
                
      return x,v
    

    def step(self, x):
      return (1+torch.sign(x))/2

      
      
    def TaOMemristiveFunc(self, Vm,MemStates):
        Gm=self.Gm*MemStates+ self.a*torch.exp(self.b*torch.sqrt(torch.abs(Vm) ) )*(1-MemStates)  
        
        im=Vm*Gm
        p=im*Vm
        deltaMemStates = self.Aa*torch.sinh(torch.tensor(Vm/self.sigma_off))*torch.exp(-torch.square(self.y_off/MemStates ))*torch.exp(1/(1+self.beta*p))*self.step(-Vm)+self.Bb*torch.sinh(Vm/self.sigma_on)*torch.exp(-torch.square(MemStates /self.y_on))*torch.exp(p/self.sigma_p)*self.step(Vm)
        
        return im, deltaMemStates  
        
    def NIBoMemristiveFunc(self, Vm):
        return Vm  
           
    def forward(self, x):
        InputCoupling=self.B(x)
        Zreshaped=self.Z.view(1,InputCoupling.shape[1],1,1).repeat(InputCoupling.shape[0],1,InputCoupling.shape[2],InputCoupling.shape[3])
        InputCoupling=InputCoupling+Zreshaped
        x=self.rescale(x)
        MemStates=self.InitMemStates.repeat( [ x.shape[0],1,1,1] )
        
        for step in range(self.IterNum):        
            Imem, deltaMem =self.TaOMemristiveFunc(self.LeakyNonLin(x),MemStates)
            Output=self.LeakyNonLin(x)
            Coupling=self.A(Output) + InputCoupling-Imem
            x=x+self.TimeStep*(-x+Coupling)           
            MemStates=MemStates+self.TimeStep*deltaMem   
        return Output





class CellNN(nn.Module):
    def __init__(self,InChannel, OutChannel):
        super(CellNN, self).__init__()
        
        self.Layer1= MeMCeNNLayer(3,16,InputShape=[16,128,256])
        self.BN1 = nn.BatchNorm2d(16)
        self.Layer2= MeMCeNNLayer(16,32,InputShape=[32,128,256])
        self.BN2 = nn.BatchNorm2d(32)
        self.Layer3= MeMCeNNLayer(32,64,InputShape=[64,128,256])
        self.BN3 = nn.BatchNorm2d(64)
        self.Layer4= MeMCeNNLayer(64,128,InputShape=[128,128,256])
        self.BN4 = nn.BatchNorm2d(128)
        self.Layer5= MeMCeNNLayer(128,OutChannel, nonlin='none',InputShape=[OutChannel,128,256])
       
        
    def forward(self, x):
    	
    	x=self.Layer1(x)
    	x=self.BN1(x)
    	x=self.Layer2(x)
    	x=self.BN2(x)
    	x=self.Layer3(x)
    	x=self.BN3(x)
    	x=self.Layer4(x)
    	x=self.BN4(x)
    	x=self.Layer5(x)
    	return x


