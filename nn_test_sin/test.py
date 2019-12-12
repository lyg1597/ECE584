import torch
import random  
import numpy as np
import matplotlib.pyplot as plt

SEED = 78901
class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1,D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, H2)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self,x):
        h1 = torch.sigmoid(self.linear1(x))
        # h2 = torch.nn.functional.relu(self.linear2(h1))
        y = self.linear2(h1)
        return y

angle_array = [i for i in range(0,540)]
data = []
label = []
data_plt = []
label_plt = []
for i in range(len(angle_array)):
    sin_val = 10*np.sin(angle_array[i]*np.pi/180)
    data_plt.append(angle_array[i])
    label_plt.append(sin_val)
    data.append([angle_array[i]])
    label.append([sin_val])

plt.plot(data_plt,label_plt)
plt.show()

data_tensor = torch.FloatTensor(data)
label_tensor = torch.FloatTensor(label)

model = TwoLayerNet(len(data[0]),100,len(label[0]))

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.99)

for t in range(100000):
    # c = list(zip(data, label))
    # random.seed(SEED)
    # random.shuffle(c)
    # data_tensor, label_tensor = zip(*c)

    # data_tensor = torch.FloatTensor(data_tensor)
    # label_tensor = torch.FloatTensor(label_tensor)
    for i in range(0,len(data),10000000000000):
        length = min(10000000000000,len(data_tensor)-i)
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(data_tensor[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, label_tensor[i:i+length])
        
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        scheduler.step()


    if(t%100 == 0):
        y_pred = model(data_tensor)

        # Compute and print loss
        loss = criterion(y_pred, label_tensor)
        print(t,loss.item())

data = torch.FloatTensor(data)
label = torch.FloatTensor(label)

y_pred = model(data)
y_pred_li= y_pred.tolist()
y_pred = [i[0] for i in y_pred_li]

y = label
y_li = y.tolist()
y = [i[0] for i in y_li]

x = data
x_li = x.tolist()
x = [i[0] for i in x_li]

plt.plot(x,y_pred)
plt.plot(x,y)
plt.show()