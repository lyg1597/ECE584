# Referenced from:
# http://www.cs.cmu.edu/~motionplanning/reading/PlanningforDynamicVeh-1.pdf
# https://ieeexplore.ieee.org/document/7225830

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

state_input = []
fn = "data_straight"
delta_const = 30
vx_const = 3
time_horizon = 30

def getIp():
    vx = vx_const
    delta = delta_const*np.pi/180
    x = [vx,delta]
    return x

def func1(vars,time):
    vy = vars[3]
    theta = vars[2]
    r = vars[4]
    ip = getIp()
    vx=ip[0]
    delta = ip[1]

    Lr = 5
    Lf = 3
    Fyf = 100
    Fyr = 100
    m = 10
    Iz = 1
    
    dx = vx*np.cos(theta) - vy*np.sin(theta)
    dy = vx*np.sin(theta) + vy*np.cos(theta)
    dtheta = r
    dvy = Fyf/m * np.cos(delta) + Fyr/m -vx*r
    dr = Lf/Iz * Fyf*np.cos(delta) - Lr/Iz *Fyr
    return [dx,dy,dtheta,dvy,dr]

timeGrid = np.arange(0,time_horizon,0.01)
# ip = np.zeros((len(timeGrid)))

#ip[300:600] = 5.0
initR = [0,0,30*np.pi/180,0,0]
fR = odeint(func1,initR,timeGrid)
fR_len = np.shape(fR)[0]
x = []
y = []
for i in range(fR_len):
    x.append(fR[i,0])
    y.append(fR[i,1])
plt.plot(x,y)
plt.show()

with open(fn+".dat",'w+') as file:   
    for i in range(np.shape(fR)[0]):
        file.write(str(timeGrid[i])+" ")
        for j in range(np.shape(fR)[1]):
            file.write(str(fR[i,j])+" ")
        file.write(str(vy_const)+" ")
        file.write(str(delta_const)+"\n")
            


