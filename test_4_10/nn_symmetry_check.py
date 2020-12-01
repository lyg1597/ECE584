import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

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

Lr = 2
Lf = 2
dt = 0.01

def func1(t,vars,args):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]%(np.pi*2)
    vr = args[0]
    delta = args[1]

    if vr > 30:
        vr = 30
    elif vr < -0:
        vr = -0

    if delta > np.pi/4: 
        delta = np.pi/4
    elif delta < -np.pi/4:
        delta = -np.pi/4

    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    dx = vr*np.cos(curr_theta+beta)
    dy = vr*np.sin(curr_theta+beta)
    dtheta = vr/Lr * np.sin(beta)
    return [dx,dy,dtheta]

def Df(t):
    dx = 12*np.exp(-0.5*t)
    dy = 4*np.exp(-0.3*t)
    dtheta = np.pi
    return dx,dy,dtheta

pos_ref = np.arange(-15,0.01,0.05)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([pos_ref[i],0,0])
    eref.append([0,0,i*0.01])

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

res_file = open('res_file','w+')

start_x = 0
start_y = 0
start_theta = 0

target_x = 0.5
target_y = 0.5
target_theta = 0

transformation_x = 0
transformation_y = 0
rotatation_theta = 0

# Perform translation
for i in range(10):
    init_x = np.random.uniform(start_x - 0.5, start_x + 0.5)
    init_y = np.random.uniform(start_y - 0.5, start_y + 0.5)
    init_theta = np.random.uniform(start_theta - np.pi/2,start_theta+np.pi/2)

    transformation_x = np.random.uniform(-10,10)
    transformation_y = np.random.uniform(-10,10)

    init_x_transformed = init_x+transformation_x
    init_y_transformed = init_y+transformation_y
    init_theta_transformed = init_theta

    target_x_transformed = target_x + transformation_x
    target_y_transformed = target_y + transformation_y
    target_theta_transformed = target_theta

    error_x = target_x - init_x
    error_y = target_y - init_y
    error_theta = (target_theta - init_theta)%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    error_x_transformed = target_x_transformed - init_x_transformed
    error_y_transformed = target_y_transformed - init_y_transformed
    error_theta_transformed = (target_theta_transformed - init_theta_transformed)%(np.pi*2)
    error_theta_cos_transformed = np.cos(error_theta_transformed)
    error_theta_sin_transformed = np.sin(error_theta_transformed)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    res = model(data)
    vr = res[0].item()
    delta = res[1].item()

    data_transformed = torch.FloatTensor([error_x_transformed, error_y_transformed, error_theta_cos_transformed, error_theta_sin_transformed])
    res_transformed = model(data_transformed)
    vr_transformed = res_transformed[0].item()
    delta_transformed = res_transformed[1].item()

    # print(error_x, error_x_transformed, error_y, error_y_transformed, error_theta, error_theta_transformed)
    # print(f"vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}")

    symmetry = (vr == vr_transformed and delta == delta_transformed)
    if symmetry:
        print("Transformation Symmetry True")
    else:
        print("Transformation Symmetry False")

    res_file.write(f"init_x {init_x}, init_y {init_y}, init_theta {init_theta}, \
target_x {target_x}, target_y {target_y}, target_theta {target_theta},\
translation, transformation_x {transformation_x}, transformation_y {transformation_y},\
init_x_transformed {init_x_transformed}, init_y_transformed {init_y_transformed}, init_theta_transformed {init_theta_transformed}, \
target_x_transformed {target_x_transformed}, target_y_transformed {target_y_transformed}, target_theta_transformed {target_theta_transformed},\
vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}, symmetry {symmetry}\n")

# Perform rotation
for i in range(10):
    init_x = np.random.uniform(start_x - 0.5, start_x + 0.5)
    init_y = np.random.uniform(start_y - 0.5, start_y + 0.5)
    init_theta = np.random.uniform(start_theta - np.pi/2,start_theta+np.pi/2)

    transformation_x = np.random.uniform(-10,10)
    transformation_y = np.random.uniform(-10,10)
    transformation_theta = np.random.uniform(-np.pi,np.pi)

    init_x_transformed = init_x*np.cos(transformation_theta)+init_y*np.sin(transformation_theta)
    init_y_transformed = -init_x*np.sin(transformation_theta)+init_y*np.cos(transformation_theta)
    init_theta_transformed = init_theta+transformation_theta

    target_x_transformed = target_x*np.cos(transformation_theta)+target_y*np.sin(transformation_theta)
    target_y_transformed = -target_x*np.sin(transformation_theta)+target_y*np.cos(transformation_theta)
    target_theta_transformed = target_theta+transformation_theta

    error_x = target_x - init_x
    error_y = target_y - init_y
    error_theta = (target_theta - init_theta)%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    error_x_transformed = target_x_transformed - init_x_transformed
    error_y_transformed = target_y_transformed - init_y_transformed
    error_theta_transformed = (target_theta_transformed - init_theta_transformed)%(np.pi*2)
    error_theta_cos_transformed = np.cos(error_theta_transformed)
    error_theta_sin_transformed = np.sin(error_theta_transformed)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    res = model(data)
    vr = res[0].item()
    delta = res[1].item()

    data_transformed = torch.FloatTensor([error_x_transformed, error_y_transformed, error_theta_cos_transformed, error_theta_sin_transformed])
    res_transformed = model(data_transformed)
    vr_transformed = res_transformed[0].item()
    delta_transformed = res_transformed[1].item()

    # print(error_x, error_x_transformed, error_y, error_y_transformed, error_theta, error_theta_transformed)
    # print(f"vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}")

    symmetry = (vr == vr_transformed and delta == delta_transformed)
    if symmetry:
        print("Transformation Symmetry True")
    else:
        print("Transformation Symmetry False")

    res_file.write(f"init_x {init_x}, init_y {init_y}, init_theta {init_theta}, \
target_x {target_x}, target_y {target_y}, target_theta {target_theta},\
rotation, transformation_theta {transformation_theta},\
init_x_transformed {init_x_transformed}, init_y_transformed {init_y_transformed}, init_theta_transformed {init_theta_transformed}, \
target_x_transformed {target_x_transformed}, target_y_transformed {target_y_transformed}, target_theta_transformed {target_theta_transformed},\
error_x {error_x}, error_y {error_y}, error_theta {error_theta},\
error_x_transformed {error_x_transformed}, error_y_transformed {error_y_transformed}, error_theta_transformed {error_theta_transformed}, \
vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}, symmetry {symmetry}\n")

# Perform mirror with respect to x axis
for i in range(10):
    init_x = np.random.uniform(start_x - 0.5, start_x + 0.5)
    init_y = np.random.uniform(start_y - 0.5, start_y + 0.5)
    init_theta = np.random.uniform(start_theta - np.pi/2,start_theta+np.pi/2)

    transformation_x = np.random.uniform(-10,10)
    transformation_y = np.random.uniform(-10,10)
    transformation_theta = np.random.uniform(-np.pi,np.pi)

    init_x_transformed = init_x
    init_y_transformed = -init_y
    init_theta_transformed = -init_theta

    target_x_transformed = target_x
    target_y_transformed = target_y
    target_theta_transformed = target_theta

    error_x = target_x - init_x
    error_y = target_y - init_y
    error_theta = (target_theta - init_theta)%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    error_x_transformed = target_x_transformed - init_x_transformed
    error_y_transformed = target_y_transformed - init_y_transformed
    error_theta_transformed = (target_theta_transformed - init_theta_transformed)%(np.pi*2)
    error_theta_cos_transformed = np.cos(error_theta_transformed)
    error_theta_sin_transformed = np.sin(error_theta_transformed)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    res = model(data)
    vr = res[0].item()
    delta = res[1].item()

    data_transformed = torch.FloatTensor([error_x_transformed, error_y_transformed, error_theta_cos_transformed, error_theta_sin_transformed])
    res_transformed = model(data_transformed)
    vr_transformed = res_transformed[0].item()
    delta_transformed = res_transformed[1].item()

    # print(error_x, error_x_transformed, error_y, error_y_transformed, error_theta, error_theta_transformed)
    # print(f"vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}")

    symmetry = (vr == vr_transformed and delta == delta_transformed)
    if symmetry:
        print("Transformation Symmetry True")
    else:
        print("Transformation Symmetry False")

    res_file.write(f"init_x {init_x}, init_y {init_y}, init_theta {init_theta}, \
target_x {target_x}, target_y {target_y}, target_theta {target_theta},\
mirror, \
init_x_transformed {init_x_transformed}, init_y_transformed {init_y_transformed}, init_theta_transformed {init_theta_transformed}, \
target_x_transformed {target_x_transformed}, target_y_transformed {target_y_transformed}, target_theta_transformed {target_theta_transformed},\
error_x {error_x}, error_y {error_y}, error_theta {error_theta},\
error_x_transformed {error_x_transformed}, error_y_transformed {error_y_transformed}, error_theta_transformed {error_theta_transformed}, \
vr {vr}, delta {delta}, vr_transformed {vr_transformed}, delta_transformed {delta_transformed}, symmetry {symmetry}\n")

res_file.close()