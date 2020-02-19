import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random

delta_const = 0
a_const = 0
v_const = 3
time_horizon = 200
Lr = 2
Lf = 2

target_x = 20
target_y = 0
target_theta = 0
target_w = 0
target_v = 0.25

k1 = 1
# k2 = 5
# k3 = 1

def getIp(curr_x, curr_y, curr_theta):
    # error_x = np.cos(curr_theta)*(target_x-curr_x)+np.sin(curr_theta)*(target_y-curr_y)
    # error_y = -np.sin(curr_theta)*(target_x-curr_x)+np.cos(curr_theta)*(target_y-curr_y)
    # error_theta = target_theta-curr_theta
    # vr = target_v*np.cos(error_theta)+k1*error_x
    # w = target_w+target_v*(k2*error_y+k3*np.sin(error_theta))

    # # if w>45*np.pi/180:
    # #     w = np.pi/4
    # # elif w<-np.pi/4:
    # #     w = -np.pi/4

    error_x = target_x - curr_x
    error_y = target_y - curr_y

    error_theta = curr_theta-np.arctan2(error_y,error_x)

    vr = 0.25
    delta = -2*error_theta

    if delta>np.pi/4:
        delta = np.pi/4
    elif delta<-np.pi/4:
        delta = -np.pi/4
    print(curr_x, curr_y, curr_theta, error_theta,delta)
    # delta = 0

    x = [vr,delta]
    return x

def func1(vars,time):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    ip = getIp(curr_x, curr_y, curr_theta)
    vr = ip[0]
    delta = ip[1]

    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    dx = vr*np.cos(curr_theta+beta)
    dy = vr*np.sin(curr_theta+beta)
    dtheta = vr/Lr * np.sin(beta)
    return [dx,dy,dtheta]

timeGrid = np.arange(0,time_horizon,0.01)
initR = [0,0,180*np.pi/180]
fR = odeint(func1,initR,timeGrid)
fR_len = np.shape(fR)[0]
x = []
y = []
for i in range(fR_len):
    x.append(fR[i,0])
    y.append(fR[i,1])
plt.plot(x,y)
plt.plot(target_x,target_y,'.')
plt.show()
