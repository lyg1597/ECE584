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
delta_const = 18
v_const = 3
time_horizon = 16
init_x = 0
init_y = 0
init_theta = 0
# 312

def getIp():
    # delta_array = [-30,0,30]
    # delta_idx = random.randint(0,2)
    v = v_const
    delta = delta_const*np.pi/180
    x = [v,delta]
    return x

def func1(vars,time):
    L = 5
    theta = vars[2]
    ip = getIp()
    v=ip[0]
    delta = ip[1]
    
    dx = v*np.cos(theta)
    dy = v*np.sin(theta)
    dtheta = v/L * np.sin(delta)/np.cos(delta) 
    return [dx,dy,dtheta]
#############################
data_straight = []
input_straight = []
output_straight = []
for i in range(4):
    data_temp = []
    with open("data_straight_"+str(int(i))+".dat") as file:
        line = file.readline()
        while line:
            line = line.split(' ')
            line = [float(i) for i in line]
            data_temp.append(line)
            line = file.readline()

    input_temp = []
    for i in range(0,len(data_temp)-1):
        input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5]])

    output_temp = []
    for i in range(1,len(data_temp)):
        temp = []
        # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
        # temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(np.pi*delta_t))%int(360))
        output_temp.append(temp)

    data_straight = data_straight + data_temp
    input_straight = input_straight + input_temp
    output_straight = output_straight + output_temp


#############################
data_pos30 = []
input_pos30 = []
output_pos30 = []
for i in range(4):
    data_temp = []
    with open("data_pos30_"+str(int(i))+".dat") as file:
        line = file.readline()
        while line:
            line = line.split(' ')
            line = [float(i) for i in line]
            data_temp.append(line)
            line = file.readline()

    input_temp = []
    for i in range(0,len(data_temp)-1):
        input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5]])

    output_temp = []
    for i in range(1,len(data_temp)):
        temp = []
        # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
        # temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(np.pi*delta_t))%int(360))
        output_temp.append(temp)

    data_pos30 = data_pos30 + data_temp
    input_pos30 = input_pos30 + input_temp
    output_pos30 = output_pos30 + output_temp

#############################
data_neg30 = []
input_neg30 = []
output_neg30 = []
for i in range(4):
    data_temp = []
    with open("data_neg30_"+str(int(i))+".dat") as file:
        line = file.readline()
        while line:
            line = line.split(' ')
            line = [float(i) for i in line]
            data_temp.append(line)
            line = file.readline()

    input_temp = []
    for i in range(0,len(data_temp)-1):
        input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5]])

    output_temp = []
    for i in range(1,len(data_temp)):
        temp = []
        # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
        # temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(np.pi*delta_t))%int(360))
        output_temp.append(temp)

    data_neg30 = data_neg30 + data_temp
    input_neg30 = input_neg30 + input_temp
    output_neg30 = output_neg30 + output_temp

#############################
data = data_straight+data_pos30+data_neg30
# input_data = input_straight+input_pos30+input_neg30
# output_data = output_straight+output_pos30+output_neg30

input_data = []
output_data = []
for i in range(len(input_straight)):
    input_data.append(input_straight[i])
    input_data.append(input_pos30[i])
    input_data.append(input_neg30[i])    

    output_data.append(output_straight[i])
    output_data.append(output_pos30[i])
    output_data.append(output_neg30[i])

# combined = list(zip(input_data,output_data))
# random.shuffle(combined)

# input_data[:], output_data[:] = zip(*combined)
trajectory = []
initial = [init_x,init_y,init_theta,v_const,delta_const]

trajectory.append(initial)

model_x = TwoLayerNet(3,100,1)
model_y = TwoLayerNet(3,100,1)
model_theta = TwoLayerNet(3,100,1)
model_x.load_state_dict(torch.load('./model_x_more_full_state'))
model_y.load_state_dict(torch.load('./model_y_more_full_state'))
model_theta.load_state_dict(torch.load('./model_theta_more_full_state'))

device = torch.device('cuda')

model_x = model_x.to(device)
model_y = model_y.to(device)
model_theta = model_theta.to(device)
temp = initial
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
    if dtheta<180:
        theta = (temp[2] + dtheta*0.01) % int(360)
    else:
       theta = (temp[2] + (dtheta-360)*0.01) % int(360)
 
    temp = [x,y,theta,3,delta_const]
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

timeGrid = np.arange(0,time_horizon,0.01)
initR = [init_x,init_y,init_theta*np.pi/180]
fR = odeint(func1,initR,timeGrid)
x_tag = []
y_tag = []
theta_tag = []
for i in range(np.shape(fR)[0]):
    x_tag.append(fR[i,0])
    y_tag.append(fR[i,1]) 
    theta_tag.append((fR[i,2]*180/np.pi)%int(360))
plt.plot(x_tag,y_tag,'bo')

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

plt.show()