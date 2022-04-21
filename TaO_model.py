import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
#original Strachan model



def eqns(t,y):
      A=param[0]
      sigma_off=param[1]
      y_off=param[2]
      beta=param[3]
      B=param[4]
      sigma_on=param[5]
      y_on=param[6]
      sigma_p=param[7]
      a=param[8]
      b=param[9]
      Gm=param[10]

      T0=param[11]
      v0=param[12]



      v=v0*np.sin(2*np.pi*t/T0)
      vm=v


      im= mem(y[0],vm,param)*vm
      # the following is the expressions for the power
      p=im*vm;

     
      out=[A*np.sinh(vm/sigma_off)*np.exp(-np.square(y_off/y[0]))*np.exp(1/(1+beta*p))*step(-vm)+B*np.sinh(vm/sigma_on)*np.exp(-np.square(y[0]/y_on))*np.exp(p/sigma_p)*step(vm), 1 ]
      return out


def mem(x1,x2,param):
      a=param[8]
      b=param[9]
      Gm=param[10]

      #x1=y state of TaO memristor
      #x2 time


      out=Gm*x1+a*np.exp(b*np.sqrt(np.abs(x2)))*(1-x1)
      return out
def step(x):
    return (1+np.sign(x))/2






period=[1, 0.1, 0.01]
for T0 in period:

      # on switching v>0
      B=1e-4
      sigma_on=.45
      y_on=.06
      sigma_p=4e-5

      # off switching v<0
      A=1e-10
      sigma_off=0.013
      y_off=0.4
      beta=500

      #parameters of memductance 
      Gm=0.025

      a=7.2e-6
      b=4.7


  
      #input parameters
      v0=0.4

      param=[A,sigma_off,y_off,beta,B,sigma_on,y_on,sigma_p,a,b,Gm,T0,v0]

      # initial condition
      yinit=0.1;
      # y1 memristor state
      # y2 time


      init=[yinit, 0]

      # simulation time
      TStart=0
      TStep=T0*1e-3
      TMax=20*T0
      #t_span=range(0,T0*1e-3,20*T0) 

       
      #options = odeset('RelTol',1e-9,'AbsTol',1e-12);
      #ode call
      #[t,y]=ode15s(@eqns,t_span,init,options,param); 
      
      
      r = scipy.integrate.ode(eqns).set_integrator('zvode', method='bdf')
      r.set_initial_value(init, 0)

      IntegaratedSeries=np.zeros(  (2,  int(TMax/TStep)+1 ) )
      Ind=0
      while r.successful() and r.t < TMax:
          IntegaratedSeries[:,Ind]=r.integrate(r.t+TStep)
          Ind+=1
    
      v=v0*np.sin(2*np.pi*IntegaratedSeries[1,:]/T0)
      vm=v
      
      #input state plot
      plt.plot(IntegaratedSeries[0,:],'b')
      plt.plot(v,'r')
      plt.show()
      
      #hysteresis plot
      im=mem(IntegaratedSeries[0,:],vm,param)*vm
      h=int(IntegaratedSeries[1,:].shape[0]/4)
      plt.plot(vm[-h:],im[-h:])
      plt.show()
      
"""
% subplot(2,1,1)
 % plot of the input
% plot(t,v)
% hold on
% xlabel('t')
% ylabel('v')
%subplot(2,1,2)
% plot of the state
% plot(t,y(:,1))
% xlabel('t')
% ylabel('x')

% figure

h=floor(length(t)/4);

% plot of the current-voltage pinched hysteresis loop
im=mem(y(:,1),vm,param).*vm;
plot(vm(end-h:end),im(end-h:end),'color',C{kk})
hold on
xlabel('$$v_m$$')
ylabel('$$i_m$$')
end
"""



