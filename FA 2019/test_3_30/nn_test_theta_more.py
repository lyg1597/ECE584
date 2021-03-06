# -*- coding: utf-8 -*-
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

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

#############################
data_straight = []
input_straight = []
output_straight = []
for i in range(8):
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
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(delta_t*np.pi))%int(360))
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
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(delta_t*np.pi))%int(360))
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
        temp.append(((data_temp[i][3]-data_temp[i-1][3])*180/(delta_t*np.pi))%int(360))
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

    output_data.append(output_straight[i])

for i in range(len(input_pos30)):
    input_data.append(input_pos30[i])
    input_data.append(input_neg30[i]) 

    output_data.append(output_pos30[i])
    output_data.append(output_neg30[i])    

x = []
y = []
theta = []
v = []
delta = []
for i in range(len(input_data)):
    # x.append(input_data[i][0])
    # y.append(input_data[i][1])
    delta.append(input_data[i][2])
    # v.append(input_data[i][3])
    # delta.append(input_data[i][4])

dx = []
dy = []
dtheta = []
for i in range(len(output_data)):
    # dx.append(output_data[i][0])
    dtheta.append(output_data[i][0])
    # dtheta.append(output_data[i][2])

plt.plot(delta,dtheta,'bo')
plt.show()    

device = torch.device('cuda')

data = torch.FloatTensor(input_data)
label = torch.FloatTensor(output_data)

data = data.to(device)
label = label.to(device)

model = TwoLayerNet(len(data[0]),100,len(label[0]))
model = model.to(device)

criterion = torch.nn.MSELoss(reduction='sum')
criterion = criterion.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
for t in range(10000):
    for i in range(0,len(data),1000):
        length = min(1000,len(data)-i)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, label[i:i+length])
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    if(t%100 == 0):
        # print(i, loss.item())
        y_pred = model(data)

        # Compute and print loss
        loss = criterion(y_pred, label)
        print(t,loss.item())
        # optimizer.zero_grad()

        # loss.backward()
        # optimizer.step()

###########################
y_pred = model(data)
y_pred = y_pred.cpu()
y_pred_li= y_pred.tolist()
y_pred = [i[0] for i in y_pred_li]

y = label.cpu()
y_li = y.tolist()
y = [i[0] for i in y_li]

x = data.cpu()
x_li = x.tolist()
x = [i[2] for i in x_li]
plt.plot(x,y_pred,'ro')
plt.plot(x,y,'bo')
plt.show()

torch.save(model.state_dict(), './model_theta_more_full_state')

print("halt")