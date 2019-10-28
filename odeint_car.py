import numpy as np
from scipy.integrate import odeint
# import matplotlib.pylab as pl
import random

state_input = []
fn = "data_straight"
delta_const = 0
v_const = 3
time_horizon = 1000

def getIp():
    # delta_array = [-30,0,30]
    # delta_idx = random.randint(0,2)
    v = v_const
    delta = delta_const*np.pi/180
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

timeGrid = np.arange(0,time_horizon,0.01)
# ip = np.zeros((len(timeGrid)))

#ip[300:600] = 5.0
initR = [0,0,0]
fR = odeint(func1,initR,timeGrid)
# print(state_input)
j = 0
with open(fn+".dat",'w+') as file:
    # for line in state_input:
    #     for i in line:
    #         file.write(str(i)+" ")
    #     file,write(str(30))
    #     file.write("\n")        
    for i in range(np.shape(fR)[0]):
        file.write(str(timeGrid[i])+" ")
        for j in range(np.shape(fR)[1]):
            file.write(str(fR[i,j])+" ")
        file.write(str(v_const)+" ")
        file.write(str(delta_const)+"\n")
        
state_input_arr = np.array(state_input)


