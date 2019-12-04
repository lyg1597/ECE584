import torch
import numpy as np
import random
import matplotlib.pyplot as plt
# [pos_x,pos_y,orientation,forward_speed,input_acceleration,input_turning]

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1,D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, H2)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self,x):
        h1 = torch.nn.functional.relu(self.linear1(x))
        # h2 = torch.nn.functional.relu(self.linear2(h1))
        y = self.linear2(h1)
        return y

delta_t = 0.01
delta_const = 30
init_v = 3
a_const = 0
time_horizon = 30
init_x = 0
init_y = 0
init_theta = 0
Lr = 5
Lf = 3

def getIp(time):
    if(time<5):
        delta = 28
    elif time>=5 and time<10:
        delta = -15
    elif time>=10 and time<15:
        delta = 16
    elif time>=15 and time<20:
        delta = -27
    else:
        delta = 16

    # delta = 0

    a = a_const
    delta_temp = delta*np.pi/180
    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta_temp)/np.cos(delta_temp))
    x = [a,delta,beta]
    return x

def func1(vars,time):
    v = vars[3]
    theta = vars[2]
    ip = getIp(time)
    a=ip[0]
    beta = ip[2]

    dx = v*np.cos(theta+beta)
    dy = v*np.sin(theta+beta)
    dtheta = v/Lr * np.sin(beta)
    dv = a 
    return [dx,dy,dtheta,dv]