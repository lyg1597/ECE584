import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import copy

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(TwoLayerNet, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, 1)
        self.control1 = torch.nn.Linear(D_in,H1)
        self.control2 = torch.nn.Linear(H1,2)

    def forward(self,x):
        # h1 = torch.nn.functional.relu(self.linear1(x))
        # # h2 = torch.nn.functional.relu(self.linear2(h1))
        # y = self.linear2(h1)

        h2 = torch.relu(self.control1(x))
        u = self.control2(h2)
        return u

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

f = open('parameter.m','w+')

weights = [model.control1.weight,model.control2.weight]
text = "weight = {"
for weight in weights:
    text = text+f"[{weight[0,0]}"
    for i in range(1,weight.shape[1]):
        text = text+f", {weight[0,i]}"
    for i in range(1,weight.shape[0]):
        text = text + ';'
        text = text + f' {weight[i,0]}'
        for j in range(1,weight.shape[1]):
            text = text + f', {weight[i,j]}'
    text = text + "] "
text = text+"};\n"
f.write(text)

biases = [model.control1.bias, model.control2.bias]
text = "bias = {"
for bias in biases:
    text = text+f"[{bias[0]}"
    for i in range(1,bias.shape[0]):
        text = text + ';'
        text = text + f' {bias[i]}'
    text = text + "] "
text = text+"};\n"
f.write(text)

f.close()

