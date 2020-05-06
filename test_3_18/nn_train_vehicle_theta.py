import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import random
import torch

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
data = []
input_data = []
output_data = []
for j in range(0,3):
    for k in range(0,12):
        with open("./data_vehicle/data"+str(int(j))+".dat") as file:
            input_temp = []
            output_temp = []
            data_temp = []
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [float(i) for i in line]
                data_temp.append(line)
                line = file.readline()

            for i in range(len(data_temp)-1):
                line = data_temp[i]
                curr_time = line[0]
                curr_x = line[1]
                curr_y = line[2]
                target_x = line[3]
                target_y = line[4]
                curr_theta = line[5]%(np.pi*2)
                curr_delta = line[6]
                curr_v = line[7]
                input_temp.append([curr_theta,curr_delta,curr_v])

                line = data_temp[i+1]
                next_time = line[0]
                next_x = line[1]
                next_y = line[2]
                target_x = line[3]
                target_y = line[4]
                next_theta = line[5]
                next_delta = line[6]
                next_v = line[7]

                x_diff = (next_x-curr_x)/delta_t
                y_diff = (next_y-curr_y)/delta_t
                theta_diff = ((next_theta-curr_theta)/delta_t)%(2*np.pi)
                if theta_diff > np.pi:
                    theta_diff = theta_diff - 2*np.pi
                output_temp.append([theta_diff])

            data = data + data_temp
            input_data = input_data + input_temp
            output_data = output_data + output_temp
            

device = torch.device('cuda')

data = torch.FloatTensor(input_data)
label = torch.FloatTensor(output_data)

data = data.to(device)
label = label.to(device)

model = TwoLayerNet(len(data[0]),100,len(label[0]))
model = model.to(device)

criterion = torch.nn.MSELoss(reduction='sum')
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)
for t in range(40000):
    for i in range(0,len(data),10000000000000):
        length = min(10000000000000,len(data)-i)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, label[i:i+length])
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        scheduler.step()

    if(t%100 == 0):
        y_pred = model(data)

        # Compute and print loss
        loss = criterion(y_pred, label)
        print(t,loss.item())

torch.save(model.state_dict(), './model_vehicle_theta')


