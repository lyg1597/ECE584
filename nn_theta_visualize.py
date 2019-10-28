import matplotlib.pyplot as plt
import torch
import numpy as np

delta_t = 0.01

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

#############################
data_straight = []
with open("data_straight.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_straight.append(line)
        line = file.readline()

input_straight = []
for i in range(0,len(data_straight)-1):
    input_straight.append([(data_straight[i][3]*180/np.pi) % int(360),data_straight[i][4],data_straight[i][5]])

output_straight = []
for i in range(1,len(data_straight)):
    temp = []
    temp.append((data_straight[i][1]-data_straight[i-1][1])/delta_t)
    temp.append((data_straight[i][2]-data_straight[i-1][2])/delta_t)
    temp.append(((data_straight[i][3]-data_straight[i-1][3])/delta_t)%int(360))
    output_straight.append(temp)

#############################
data_pos30 = []
with open("data_pos30.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_pos30.append(line)
        line = file.readline()

input_pos30 = []
for i in range(0,len(data_pos30)-1):
    input_pos30.append([(data_pos30[i][3]*180/np.pi) % int(360),data_pos30[i][4],data_pos30[i][5]])

output_pos30 = []
for i in range(1,len(data_pos30)):
    temp = []
    temp.append((data_pos30[i][1]-data_pos30[i-1][1])/delta_t)
    temp.append((data_pos30[i][2]-data_pos30[i-1][2])/delta_t)
    temp.append(((data_pos30[i][3]-data_pos30[i-1][3])/delta_t)%int(360))
    output_pos30.append(temp)

#############################
data_neg30 = []
with open("data_neg30.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_neg30.append(line)
        line = file.readline()

input_neg30 = []
for i in range(0,len(data_neg30)-1):
    input_neg30.append([(data_neg30[i][3]*180/np.pi) % int(360),data_neg30[i][4],data_neg30[i][5]])

output_neg30 = []
for i in range(1,len(data_neg30)):
    temp = []
    temp.append((data_neg30[i][1]-data_neg30[i-1][1])/delta_t)
    temp.append((data_neg30[i][2]-data_neg30[i-1][2])/delta_t)
    temp.append(((data_neg30[i][3]-data_neg30[i-1][3])/delta_t)%int(360))
    output_neg30.append(temp)

input_data = input_straight+input_neg30+input_pos30
output_data = output_straight+output_neg30+output_pos30

model = TwoLayerNet(len(input_data[0]),10,10,1)
#############################
data = torch.FloatTensor(input_data)
label = []
for i in range(len(output_data)):
    label.append([output_data[2]])
#############################



model.load_state_dict(torch.load('./model_theta'))

y_pred = model(data)
y_pred_li= y_pred.tolist()
y_pred = [i[0] for i in y_pred_li]

y = [i[0] for i in label]
plt.plot(y_pred,c='r')
plt.plot(y,c='b')
plt.show()

