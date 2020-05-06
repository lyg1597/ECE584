# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint

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
delta_const = -30*np.pi/180
init_v = 0.25
# a_const = 0
time_horizon = 50
init_x = 0
init_y = 0
init_theta = 0
Lr = 2
Lf = 2
# 312

def getIp():
    delta = delta_const
    x = [delta]
    return x

def func1(vars,time):
    v = 0.25
    theta = vars[2]
    ip = getIp()
    delta = ip[0]
    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    
    dx = v*np.cos(theta+beta)
    dy = v*np.sin(theta+beta)
    dtheta = v/Lr * np.sin(beta)
    return [dx,dy,dtheta]

##########################################################

# input_data = []
# output_data = []
# for i in range(len(input_straight)):
#     input_data.append(input_straight[i])
#     input_data.append(input_pos30[i])
#     input_data.append(input_neg30[i])    

#     output_data.append(output_straight[i])
#     output_data.append(output_pos30[i])
#     output_data.append(output_neg30[i])

# combined = list(zip(input_data,output_data))
# random.shuffle(combined)

# input_data[:], output_data[:] = zip(*combined)
trajectory = []
initial = [init_x,init_y,init_theta,delta_const,0.25]

trajectory.append(initial)

model_x = TwoLayerNet(3,100,1)
model_y = TwoLayerNet(3,100,1)
model_theta = TwoLayerNet(3,100,1)

model_x.load_state_dict(torch.load('./model_vehicle_x'))
model_y.load_state_dict(torch.load('./model_vehicle_y'))
model_theta.load_state_dict(torch.load('./model_vehicle_theta'))

device = torch.device('cuda')

model_x = model_x.to(device)
model_y = model_y.to(device)
model_theta = model_theta.to(device)

temp = initial
x_list = []
y_list = []
theta_list = []

for i in range(int(time_horizon/delta_t)):
    data = [temp[2:5]]
    x_tensor = torch.FloatTensor(data)
    x_tensor = x_tensor.to(device)
    dx_tensor = model_x(x_tensor)
    dx_tensor.cpu()
    dx = dx_tensor.data.tolist()[0][0]

    y_tensor = torch.FloatTensor(data)
    y_tensor = y_tensor.to(device)
    dy_tensor = model_y(y_tensor)
    dy_tensor.cpu()    
    dy = dy_tensor.data.tolist()[0][0]

    theta_tensor = torch.FloatTensor(data)
    theta_tensor = theta_tensor.to(device)
    dtheta_tensor = model_theta(theta_tensor)
    dtheta_tensor.cpu()
    dtheta = dtheta_tensor.data.tolist()[0][0]

    x = temp[0] + dx*0.01
    y = temp[1] + dy*0.01 
 
    # if dy>0:
    #     print("stop")
    theta = (temp[2] + dtheta*0.01) % (2*np.pi)
    # if theta > np.pi:
    #     theta = theta - 2*np.pi
    x_list.append(x)
    y_list.append(y)
    theta_list.append(theta)
    temp = [x,y,theta,delta_const,0.25]
    trajectory.append(temp)

# data = torch.FloatTensor(input_data)
# label = torch.FloatTensor(output_data)
# data = data.to(device)
# label = label.to(device)

###########################
x = []
y = []
theta = []
for i in range(len(trajectory)):
    x.append(trajectory[i][0])
    y.append(trajectory[i][1])
    theta.append(trajectory[i][2])
plt.plot(x,y,'ro')

timeGrid = np.arange(0,time_horizon+0.005,0.01)
initR = [init_x,init_y,init_theta]
fR = odeint(func1,initR,timeGrid)
x_tag = []
y_tag = []
theta_tag = []
for i in range(np.shape(fR)[0]):
    x_tag.append(fR[i,0])
    y_tag.append(fR[i,1]) 
    theta_tag.append(fR[i,2]%(2*np.pi))
plt.plot(x_tag,y_tag,'bo')
plt.show()

plt.plot(x,'.')
plt.plot(x_tag,'.')
plt.show()
plt.plot(y,'.')
plt.plot(y_tag,'.')
plt.show()
plt.plot(theta,'.')
plt.plot(theta_tag,'.')
plt.show()

# y_pred = model_theta(data)
# y_pred = y_pred.cpu()
# y_pred_li= y_pred.tolist()
# y_pred = [i[0] for i in y_pred_li]

# y = label.cpu()
# y_li = y.tolist()
# y = [i[0] for i in y_li]

# x = data.cpu()
# x_li = x.tolist()
# x = [i[2] for i in x_li]
# plt.plot(x,y_pred,'ro')
# plt.plot(x,y,'bo')
