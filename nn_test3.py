# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1,H2, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y1 = torch.nn.functional.relu(self.linear1(x))
        y2 = torch.nn.functional.relu(self.linear2(y1))
        y_pred = self.linear3(y2)
        return y_pred

device = torch.device("cuda")
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H1,H2, D_out =1, 100,100, 1

# Create random Tensors to hold inputs and outputs
x = np.arange(0,20,0.001)
y = np.sin(x)

x_li = []
y_li = []
for i in range(len(x)):
    x_li.append([x[i]])
    y_li.append([y[i]])
    

x = torch.FloatTensor(x_li)
y = torch.FloatTensor(y_li)

x = x.to(device)
y = y.to(device)

# Construct our model by instantiating the class defined above
model = TwoLayerNet(D_in, H1,H2, D_out)
model = model.to(device)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.L1Loss(reduction='sum')
criterion = criterion.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-7)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-9)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-10)
for t in range(50000):
    for i in range(0,len(y_li),1000000000):
        length = min(1000000000,len(y_li)-i)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x[i:i+length])

        # Compute and print loss
        loss = criterion(y_pred, y[i:i+length])
        if t % 100 == 0:
            print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

y_pred = model(x)
y_pred = y_pred.cpu()
y_pred_li= y_pred.tolist()
y_pred = [i[0] for i in y_pred_li]

y = y.cpu()
y_li = y.tolist()
y = [i[0] for i in y_li]
plt.plot(y_pred,c='r')
plt.plot(y,c='b')
plt.show()

torch.save(model.state_dict(), './model_sine')
