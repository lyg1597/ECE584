import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import random

fn = "./data_vehicle/data"

time_horizon = 1000
delta_const = 0
a_const = 0
v_const = 0.25
Lr = 2
Lf = 2

target_x = 20
target_y = 0
target_theta = 0
target_w = 0
target_v = 1

k1 = -4

def getIp(curr_x, curr_y, curr_theta):

    # error_x = target_x - curr_x
    # error_y = target_y - curr_y

    # error_theta = curr_theta-np.arctan2(error_y,error_x)

    # vr = 0.25
    # delta = k1*error_theta

    # if delta>np.pi/4:
    #     delta = np.pi/4
    # elif delta<-np.pi/4:
    #     delta = -np.pi/4
    # # delta = 0

    vr = v_const
    delta = delta_const

    x = [vr,delta]
    return x

def func1(t,vars):
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

for i in range(0,11):
    delta_const = (-50+i*10)*np.pi/180
    print(i)
    x_init = 0
    y_init = 0
    theta_init = 0
    initR = [y_init,x_init,theta_init]
    r = ode(func1)
    r.set_initial_value(initR)
    dt = 0.01

    x = []
    y = []
    time = []
    theta = []
    delta = []
    v = []

    curr_x = x_init
    curr_y = y_init
    curr_theta = theta_init
    while r.t<time_horizon:
        x.append(curr_x)
        y.append(curr_y)
        theta.append(curr_theta)
        time.append(r.t)
        control = getIp(curr_x, curr_y, curr_theta)
        delta.append(control[1])
        v.append(control[0])

        val = r.integrate(r.t + dt)

        curr_x = val[0]
        curr_y = val[1]
        curr_theta = val[2]

    plt.plot(x,y)
    plt.show()

    with open(fn+str(i)+".dat",'w+') as file:   
        for j in range(len(time)):
            file.write(str(time[j])+" ")
            file.write(str(x[j])+' ')            
            file.write(str(y[j])+' ')
            file.write(str(target_x)+' ')
            file.write(str(target_y)+' ')
            file.write(str(theta[j])+' ')
            file.write(str(delta[j])+' ')
            file.write(str(v[j])+'\n')     
       

