import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

def vehicle_dynamics(t,vars,args):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]%(np.pi*2)
    vr = args[0]
    delta = args[1]

    if vr > 100:
        vr = 100
    elif vr < -0:
        vr = -0

    if delta > np.pi/3: 
        delta = np.pi/3
    elif delta < -np.pi/3:
        delta = -np.pi/3

    # beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    # dx = vr*np.cos(curr_theta+beta)
    # dy = vr*np.sin(curr_theta+beta)
    # dtheta = vr/Lr * np.sin(beta)
    dx = vr*np.cos(curr_theta+delta)
    dy = vr*np.sin(curr_theta+delta)
    dtheta = delta
    return [dx,dy,dtheta]

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(TwoLayerNet, self).__init__()
        self.control1 = torch.nn.Linear(D_in,H1)
        self.control2 = torch.nn.Linear(H1,2)

    def forward(self,x):
        h2 = torch.relu(self.control1(x))
        u = self.control2(h2)
        return u

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))
