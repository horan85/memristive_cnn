
function hp_TaO_model;
clc,clear all,close all
C = {'k','b','r','g','y',[.5 .6 .7],[.8 .2 .6]} % cell array of colours

period=[1 0.1 0.01];
for kk=1:length(period)

% on switching v>0
B=1e-4;
sigma_on=.45;
y_on=.06;
sigma_p=4e-5;


% off switching v<0
 A=1e-10;
 
  sigma_off=0.013;
  y_off=.4;
  beta=500;

% parameters of memductance 
Gm=.025;

a=7.2e-6;
b=4.7;


  
%input parameters
v0=0.4;
T0=period(kk);

param=[A,sigma_off,y_off,beta,B,sigma_on,y_on,sigma_p,a,b,Gm,T0,v0];

% initial condition
yinit=0.1;
% y1 memristor state
% y2 time


init=[yinit;0];

% simulation time
t_span=0:T0*1e-3:20*T0; 

 
options = odeset('RelTol',1e-9,'AbsTol',1e-12);
% ode call
[t,y]=ode15s(@eqns,t_span,init,options,param); 

% post processing
v=v0*sin(2*pi*t/T0);
vm=v;

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





function out = eqns(t,y,param);

A=param(1);
sigma_off=param(2);
y_off=param(3);
beta=param(4);
B=param(5);
sigma_on=param(6);
y_on=param(7);
sigma_p=param(8);
a=param(9);
b=param(10);
Gm=param(11);

T0=param(12);
v0=param(13);



v=v0*sin(2*pi*t/T0);
vm=v;


im=mem(y(1),vm,param).*vm;
% the following is the expressions for the power
p=im*vm;



out=[A*sinh(vm/sigma_off)*exp(-(y_off/y(1))^2)*exp(1/(1+beta*p))*step(-vm)+B*sinh(vm/sigma_on)*exp(-(y(1)/y_on)^2)*exp(p/sigma_p)*step(vm);
    1 ;
    ];



function out=mem(x1,x2,param);
a=param(9);
b=param(10);
Gm=param(11);

% x1=y state of TaO memristor
% x2 time


out=Gm*x1+a*exp(b*sqrt(abs(x2))).*(1-x1);

function out=step(x);
         out=(1+sign(x))/2;




