import numpy as np
from scipy.integrate import odeint
import pylab as pl
import random


def getIp():
        v = 3
        delta = 30*np.pi/180
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
        return [dx,dy,dtheta]

timeGrid = np.arange(0,1000,0.01)
# ip = np.zeros((len(timeGrid)))

#ip[300:600] = 5.0
initR = [0,0,0]
fR = odeint(func1,initR,timeGrid)

pl.figure()
# pl.plot(timeGrid,ip,'k-',label='ip')
# pl.plot(timeGrid,fR[:,0],'b-',label='x')
# pl.plot(timeGrid,fR[:,1],'r-',label='y')
pl.plot(fR[:,0],fR[:,1],'r-',label='y')
pl.legend()
pl.show()