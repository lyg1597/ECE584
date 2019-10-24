# -*- coding: utf-8 -*-
import torch
import numpy as np

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1,H2, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)
        # self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self,x):
        h1 = torch.nn.functional.relu(self.linear1(x))
        # h2 = torch.nn.functional.relu(self.linear2(h1))
        y = self.linear2(h1)
        return y


delta_t = 0.01

#############################
data_straight = []
with open("data_straight") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_straight.append(line)
        line = file.readline()

input_straight = []
for i in range(0,len(data_straight)-1):
    input_straight.append([data_straight[i][3]%int(360),data_straight[i][4],data_straight[i][5]])

output_straight = []
for i in range(1,len(data_straight)):
    temp = []
    temp.append((data_straight[i][1]-data_straight[i-1][1])/delta_t)
    temp.append((data_straight[i][2]-data_straight[i-1][2])/delta_t)
    temp.append(((data_straight[i][3]-data_straight[i-1][3])/delta_t)%int(360))
    output_straight.append(temp)

#############################
data_pos30 = []
with open("data_pos30") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_pos30.append(line)
        line = file.readline()

input_pos30 = []
for i in range(0,len(data_pos30)-1):
    input_pos30.append([data_pos30[i][3]%int(360),data_pos30[i][4],data_pos30[i][5]])

output_pos30 = []
for i in range(1,len(data_pos30)):
    temp = []
    # temp.append((data_pos30[i][1]-data_pos30[i-1][1])/delta_t)
    # temp.append((data_pos30[i][2]-data_pos30[i-1][2])/delta_t)
    temp.append(((data_pos30[i][3]-data_pos30[i-1][3])/delta_t)%int(360))
    output_pos30.append(temp)

#############################
data_neg30 = []
with open("data_neg30") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_neg30.append(line)
        line = file.readline()

input_neg30 = []
for i in range(0,len(data_neg30)-1):
    input_neg30.append([data_neg30[i][3]%int(360),data_neg30[i][4],data_neg30[i][5]])

output_neg30 = []
for i in range(1,len(data_neg30)):
    temp = []
    temp.append((data_neg30[i][1]-data_neg30[i-1][1])/delta_t)
    temp.append((data_neg30[i][2]-data_neg30[i-1][2])/delta_t)
    temp.append(((data_neg30[i][3]-data_neg30[i-1][3])/delta_t)%int(360))
    output_neg30.append(temp)

#############################
data = data_straight+data_pos30+data_neg30
input_data = input_straight+input_pos30+input_neg30
output_data = output_straight+output_pos30+output_neg30

with open('input_data','w+') as fd:
    for line in input_data:
        fd.write(str(line[0]))
        fd.write(' ')
        fd.write(str(line[1]))
        fd.write(' ')
        fd.write(str(line[2]))
        fd.write('\n')

with open('output_data','w+') as fd:
    for line in output_data:
        fd.write(str(line[0]))
        # fd.write(' ')
        # fd.write(str(line[1]))
        # fd.write(' ')
        # fd.write(str(line[2]))
        fd.write('\n')

device = torch.device('cuda')

data = torch.FloatTensor(input_pos30)
label = torch.FloatTensor(output_pos30)

data = data.to(device)
label = label.to(device)

model = TwoLayerNet(len(input_data[0]),10,10,len(output_pos30[0]))
model = model.to(device)

criterion = torch.nn.MSELoss(reduction='sum')
criterion = criterion.to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
# # optimizer.to(device)

# for t in range(5000):
#     # Forward pass: Compute predicted y by passing x to the model
#     y_pred = model(data)

#     # Compute and print loss
#     loss = criterion(y_pred, label)
#     if t % 100 == 0:
#         print(t, loss.item())

#     # Zero gradients, perform a backward pass, and update the weights.
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(1000):
    for i in range(0,len(data),100):
        length = min(100,len(data)-i)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, label[i:i+length])
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # print(i, loss.item())
    y_pred = model(data)

        # Compute and print loss
    loss = criterion(y_pred, label)
    print(t, loss.item())

print("halt")