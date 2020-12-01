import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import yaml
from torch.autograd import Variable
import sys

class FFNNC(torch.nn.Module):
    def __init__(self,D_in = 6, D_out = 8):
        super(FFNNC, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, 20)
        self.layer2 = torch.nn.Linear(20,20)
        self.layer3 = torch.nn.Linear(20,D_out)

    def forward(self,x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = self.layer3(x)
        return x

def convert_model(fn):
    f = open(fn)
    model_yaml = yaml.safe_load(f)
    bias1 = np.array(model_yaml['offsets'][1])
    bias2 = np.array(model_yaml['offsets'][2])
    bias3 = np.array(model_yaml['offsets'][3])
    weight1 = np.array(model_yaml['weights'][1])
    weight2 = np.array(model_yaml['weights'][2])
    weight3 = np.array(model_yaml['weights'][3])

    bias1 = torch.FloatTensor(bias1)
    bias2 = torch.FloatTensor(bias2)
    bias3 = torch.FloatTensor(bias3)
    weight1 = torch.FloatTensor(weight1)
    weight2 = torch.FloatTensor(weight2)
    weight3 = torch.FloatTensor(weight3)

    controller = FFNNC()
    controller.layer1.weight = torch.nn.Parameter(weight1)
    controller.layer2.weight = torch.nn.Parameter(weight2)
    controller.layer3.weight = torch.nn.Parameter(weight3)
    controller.layer1.bias = torch.nn.Parameter(bias1)
    controller.layer2.bias = torch.nn.Parameter(bias2)
    controller.layer3.bias = torch.nn.Parameter(bias3)        

    dummy_input = Variable(torch.FloatTensor([0,0,0,0,0,0]))
    torch.onnx.export(controller,dummy_input,'./tanh20x20.onnx')

if __name__ == "__main__":
    fn = sys.argv[1]
    convert_model(fn)