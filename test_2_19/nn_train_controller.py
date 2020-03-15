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

data = []
input_data = []
output_data = []
for j in range(0,2):
    for i in range(0,12):
        with open("./data/data"+str(int(j))+"_"+str(int(i))+".dat") as file:
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [float(i) for i in line]
                data.append(line)
                curr_x = line[1]
                curr_y = line[2]
                target_x = line[3]
                target_y = line[4]
                curr_theta = line[5]
                delta = line[6]
                dist = np.sqrt((curr_x-target_x)**2+(curr_y-target_y)**2)
                error_x = target_x-curr_x
                error_y = target_y-curr_y
                angle = (curr_theta - np.arctan2(error_y,error_x))%(np.pi*2)
                input_data.append([angle,dist])
                output_data.append([delta])
                line = file.readline()

device = torch.device('cuda')
data = torch.FloatTensor(input_data)
label = torch.FloatTensor(output_data)
data = data.to(device)
label = label.to(device)
model = TwoLayerNet(len(data[0]),100,len(label[0]))
model = model.to(device)
criterion = torch.nn.MSELoss(reduction='sum')
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)
for t in range(100000):
    for i in range(0,len(data),1000000000000):
        length = min(1000000000000,len(data)-i)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, label[i:i+length])
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    if(t%100 == 0):
        y_pred = model(data)

        # Compute and print loss
        loss = criterion(y_pred, label)
        print(t,loss.item())

torch.save(model.state_dict(), './model_controller')

print("halt")