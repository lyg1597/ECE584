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

fn = "./data/data"

time_horizon = 1000
delta_const = 0
a_const = 0
v_const = 3
Lr = 2
Lf = 2

target_x = 17
target_y = 0
target_theta = 0
target_w = 0
target_v = 1

k1 = -4

model = TwoLayerNet(2,100,1)
model.load_state_dict(torch.load('./model_controller'))

model_x = TwoLayerNet(3,100,1)
model_x.load_state_dict(torch.load('./model_vehicle_x'))
model_y = TwoLayerNet(3,100,1)
model_y.load_state_dict(torch.load('./model_vehicle_y'))
model_theta = TwoLayerNet(3,100,1)
model_theta.load_state_dict(torch.load('./model_vehicle_theta'))

def getIp(curr_x, curr_y, curr_theta):

    error_x = target_x - curr_x
    error_y = target_y - curr_y

    error_theta = curr_theta-np.arctan2(error_y,error_x)
    error_pos = np.sqrt((target_x-curr_x)**2+(target_y-curr_y)**2)

    vr = 0.25
    # delta = k1*error_theta

    input_tensor = torch.FloatTensor([[error_theta,error_pos]])
    output_tensor = model(input_tensor)

    delta = output_tensor.data.tolist()[0][0]

    if delta>np.pi/4:
        delta = np.pi/4
    elif delta<-np.pi/4:
        delta = -np.pi/4
    # delta = 0

    x = [vr,delta]
    return x

def func1(t,vars):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]
    ip = getIp(curr_x, curr_y, curr_theta)
    vr = ip[0]
    delta = ip[1]

    # beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    # dx = vr*np.cos(curr_theta+beta)
    # dy = vr*np.sin(curr_theta+beta)
    # dtheta = vr/Lr * np.sin(beta)

    input_tensor = torch.FloatTensor([[curr_theta,delta,vr]])
    dx_tensor = model_x(input_tensor)
    dy_tensor = model_y(input_tensor)
    dtheta_tensor = model_theta(input_tensor)

    dx = dx_tensor.data.tolist()[0][0]
    dy = dy_tensor.data.tolist()[0][0]
    dtheta = dtheta_tensor.data.tolist()[0][0]
    return [dx,dy,dtheta]


x_init = 0
y_init = 0
theta_init = 160*np.pi/180
initR = [y_init,x_init,theta_init]
r = ode(func1)
r.set_initial_value(initR)
dt = 0.01

x = []
y = []
time = []
theta = []

curr_x = x_init
curr_y = y_init
curr_theta = theta_init
while np.sqrt((target_x-curr_x)**2+(target_y-curr_y)**2) > 0.01 and r.t<time_horizon:
    x.append(curr_x)
    y.append(curr_y)
    theta.append(curr_theta)
    time.append(r.t)

    val = r.integrate(r.t + dt)

    curr_x = val[0]
    curr_y = val[1]
    curr_theta = val[2]

plt.plot(x,y)
plt.plot(target_x,target_y,'.')
plt.show()


