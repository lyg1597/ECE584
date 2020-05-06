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

        h2 = torch.nn.functional.relu(self.control1(x))
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
    elif vr < -30:
        vr = -30

    if delta > np.pi/4: 
        delta = np.pi/4
    elif delta < -np.pi/4:
        delta = -np.pi/4

    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    dx = vr*np.cos(curr_theta+beta)
    dy = vr*np.sin(curr_theta+beta)
    dtheta = vr/Lr * np.sin(beta)
    return [dx,dy,dtheta]

pos_ref = np.arange(-15,0.01,0.05)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([pos_ref[i],0,0])
    eref.append([0,0,i*0.01])

model = TwoLayerNet(3,100)
model.load_state_dict(torch.load('./model_controller'))

for i in range(1000):
    x_init = np.random.uniform(-1-0.01,1-0.01)
    y_init = np.random.uniform(-1,1)
    theta_init = np.random.uniform(-np.pi/2,np.pi/2)

    trajectory = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])

    error_x = 0-x_init
    error_y = 0-y_init
    error_theta = (0-theta_init)%(np.pi*2)

    data = torch.FloatTensor([error_x,error_y,error_theta])
    u = model(data)
    vr = u[0].item()
    delta = u[1].item()

    r.set_f_params([vr,delta])
    val = r.integrate(r.t+0.01)

    trajectory.append([r.t,val[0],val[1],val[2]])
    error_pos = np.sqrt((val[0]-0)**2+(val[1]-0)**2)


    # x = []
    # y = []
    # for i in range(len(trajectory)):
    #     x.append(trajectory[i][1])
    #     y.append(trajectory[i][2])
    x = val[0]
    y = val[1]

    plt.plot([x_init,x],[y_init,y],'b')
    plt.plot(x_init,y_init,'g.')
    plt.plot(x,y,'r.')

plt.plot(0,0,'y.')
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)

plt.show()