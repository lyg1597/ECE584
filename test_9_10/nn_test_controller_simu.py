import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import copy

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(TwoLayerNet, self).__init__()
        self.control1 = torch.nn.Linear(D_in,H1)
        self.control2 = torch.nn.Linear(H1,2)

    def forward(self,x):
        h2 = torch.relu(self.control1(x))
        u = self.control2(h2)
        return u

def vehicle_dynamics(t,vars,args):
    # Dynamics of the vehicle
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

    dx = vr*np.cos(curr_theta+delta)
    dy = vr*np.sin(curr_theta+delta)
    dtheta = vr/4*np.sin(delta)
    drefy = 50
    return [dx,dy,dtheta,drefy]

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)

data = []
x_init = 0
y_init = 0
theta_init = 0
for i in range(30):
    # Run the model for 30 times with different initial condition
    x_init = np.random.uniform(-1.5,1.5)
    y_init = np.random.uniform(-16,-14)
    theta_init = np.pi/2

    trajectory = [[0,x_init,y_init,theta_init,-15+0.5]]
    r = ode(vehicle_dynamics)
    r.set_initial_value([x_init,y_init,theta_init,-15+0.5])
    time = [0]
    ref = [[0,-15,np.pi/2],[0,-15+0.5,np.pi/2]]
    # Run the controller and vehicle for 30 steps.
    for i in range(30):
        # Calculate error
        error_x = ref[i+1][0]-trajectory[i][1]
        error_y = ref[i+1][1]-trajectory[i][2]
        error_theta = (ref[i+1][2]-trajectory[i][3])%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sin(error_theta)

        # Get control input
        data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
        u = model(data)
        vr = u[0].item()
        delta = u[1].item()

        # Run plant
        r.set_f_params([vr,delta])
        val = r.integrate(r.t+0.01)

        trajectory.append([r.t,val[0],val[1],val[2]])
        ref.append([0,val[3],np.pi/2])

        time.append(r.t)
        print(i,vr,delta,val[0],val[1],val[2],val[3])

    print(x_init,y_init,theta_init*180/np.pi)

    x = []
    y = []
    theta = []
    ref_x = []
    ref_y = []
    ref_theta = []
    for i in range(len(trajectory)):
        x.append(trajectory[i][1])
        y.append(trajectory[i][2])
        theta.append(trajectory[i][3])
        ref_x.append(ref[i][0])
        ref_y.append(ref[i][1])
        ref_theta.append(ref[i][2])

    plt.figure(1)
    plt.plot(x,y,'b')
    plt.plot(x_init,y_init,'r.')
    plt.plot(ref_x,ref_y,'g--')
    plt.plot(ref_x[0],ref_y[0],'y.')
    plt.plot(ref_x[-1],ref_y[-1],'y.')
    plt.title('y vs x')

    plt.figure(2)
    plt.plot(time,x,'b')
    plt.title('x vs time')
    
    plt.figure(3)
    plt.plot(time,y,'b')
    plt.title('y vs time')
    
    plt.figure(4)
    plt.plot(time,theta,'b')
    plt.title('theta vs time')

plt.figure(2)

plt.figure(3)

plt.figure(4)

plt.figure(1)
plt.show()
