import numpy as np
from scipy.integrate import odeint
import pylab as pl
import random

state_input = []

def getIp():
    delta_array = [-30,0,30]
    delta_idx = random.randint(0,2)
    v = 3
    delta = delta_array[delta_idx]*np.pi/180
    x = [v,delta]
    return x

def func1(vars,time):
    # ip = [0,5]
    # idx = random.randint(0,1)
    # print(idx)
    # leak = -0.3
    # a = d = 0.1
    # b = c = -0.2
    # dxbydt = leak*Rates[0]+ a*Rates[0]+b/2.*Rates[1]+ip[idx]
    # dybydt = leak*Rates[1]+ c*Rates[0]+d*Rates[1]+ip[idx]
    # return [dxbydt, dybydt]
    # dvx = 
    # dvy = 
    # dr = 
    L = 5
    theta = vars[2]
    ip = getIp()
    v=ip[0]
    delta = ip[1]
    
    dx = v*np.cos(theta)
    dy = v*np.sin(theta)
    dtheta = v/L * np.sin(delta)/np.cos(delta) 
    state_input.append([time,vars[0],vars[1],vars[2],v,delta])
    return [dx,dy,dtheta]

timeGrid = np.arange(0,1,0.01)
# ip = np.zeros((len(timeGrid)))

#ip[300:600] = 5.0
initR = [0,0,0]
fR = odeint(func1,initR,timeGrid)
print(state_input)
j = 0
with open('data','w+') as file:
    for line in state_input:
        for i in line:
            file.write(str(i)+" ")
        file.write("\n")        

state_input = np.array(state_input)

pl.figure()
# pl.plot(timeGrid,ip,'k-',label='ip')
pl.plot(timeGrid,fR[:,0],'b-',label='x')
pl.plot(timeGrid,fR[:,1],'r-',label='y')
pl.plot(timeGrid,fR[:,2]*np.pi/180,'g-',label='theta')

pl.legend()
# pl.show(block=False)
pl.show()

pl.figure()
pl.plot(fR[:,0],fR[:,1],'r-',label='y')
pl.plot(state_input[:,1],state_input[:,2],'b-',label='y')
pl.legend()
pl.show(block=False)
pl.show()
