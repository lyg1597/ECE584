# Referenced from:
# https://ieeexplore.ieee.org/document/7225830
# [pos_x,pos_y,orientation,forward_speed,input_acceleration,input_turning]

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

state_input = []
fn = "./data/data_neg30_"
delta_const = -30
a_const = -1.2
v_const = 3
time_horizon = 120
Lr = 5
Lf = 3

def getIp():
    a = a_const
    delta = delta_const*np.pi/180
    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    x = [a,delta,beta]
    return x

def func1(vars,time):
    v = vars[3]
    theta = vars[2]
    ip = getIp()
    a=ip[0]
    beta = ip[2]

    dx = v*np.cos(theta+beta)
    dy = v*np.sin(theta+beta)
    dtheta = v/Lr * np.sin(beta)
    dv = a 
    return [dx,dy,dtheta,dv]

timeGrid = np.arange(0,time_horizon,0.01)
# ip = np.zeros((len(timeGrid)))

for k in range(-3,4):
    a_const = k*0.4
    initR = [0,0,30*np.pi/180,3]
    fR = odeint(func1,initR,timeGrid)
    fR_len = np.shape(fR)[0]
    x = []
    y = []
    for i in range(fR_len):
        x.append(fR[i,0])
        y.append(fR[i,1])
    plt.plot(x,y)
    plt.show()

    with open(fn+str(int(k))+".dat",'w+') as file:   
        for i in range(np.shape(fR)[0]):
            file.write(str(timeGrid[i])+" ")
            for j in range(np.shape(fR)[1]):
                file.write(str(fR[i,j])+" ")
            file.write(str(a_const)+" ")
            file.write(str(delta_const)+"\n")
            