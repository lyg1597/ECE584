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

Lr = 2
Lf = 2
dt = 0.01

def func1(t,vars,args):
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
    return [dx,dy,dtheta]

pos_ref = np.arange(-12,0,0.4)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([0,pos_ref[i],np.pi/2])
    eref.append([0,0,i*0.01])

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)

x_start = []
y_start = []
x_end = []
y_end = []
data = []
label = []
vr_list = []
delta_list = []
x_init = 0
y_init = 0
theta_init = 0
ref_origin = copy.deepcopy(ref)
for i in range(1):
    x_init = 3.67529 - 4
    y_init = -3.9847 - 8
    theta_init = 0.00705

    trajectory = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])
    time = [0]
    for i in range(len(ref)-1):
        error_x = ref[i+1][0]-trajectory[i][1]
        error_y = ref[i+1][1]-trajectory[i][2]
        error_theta = (ref[i+1][2]-trajectory[i][3])%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sin(error_theta)
        x_start.append(trajectory[i][1]-ref[i+1][0])
        y_start.append(trajectory[i][2]-ref[i+1][1])

        data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
        u = model(data)
        vr = u[0].item()
        delta = u[1].item()

        r.set_f_params([vr,delta])
        val = r.integrate(r.t+0.01)

        # if vr > 100:
        #     vr = 100
        # elif vr < -0:
        #     vr = -0

        # if delta > np.pi/3: 
        #     delta = np.pi/3
        # elif delta < -np.pi/3:
        #     delta = -np.pi/3

        # val = [0,0,0]

        # val[0] = trajectory[i][1] + 0.01*vr*np.cos(trajectory[i][3]+delta)
        # val[1] = trajectory[i][2] + 0.01*vr*np.sin(trajectory[i][3]+delta)
        # val[2] = trajectory[i][3] + 0.01*delta

        x_end.append(val[0]-ref[i+1][0])
        y_end.append(val[1]-ref[i+1][1])
        trajectory.append([r.t,val[0],val[1],val[2]])
        error_pos = np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
        
        time.append(r.t)
        print(i,vr,delta,error_pos,error_x,error_y,error_theta*180/np.pi)
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
    # plt.plot(x,y,'g.')
    plt.plot(x_init,y_init,'r.')
    plt.plot(ref_x,ref_y,'g--')
    plt.plot(ref_x[0],ref_y[0],'y.')
    plt.plot(ref_x[-1],ref_y[-1],'y.')


    plt.figure(2)
    plt.plot(time,x,'b')
    # plt.plot(time,x,'g.')
    
    plt.figure(3)
    plt.plot(time,y,'b')
    # plt.plot(time,y,'g.')
    
    plt.figure(4)
    plt.plot(time,theta,'b')
    # plt.plot(time,theta,'g.')
    
x_up = []
x_low = []
# y_up = []
# y_low = []
# theta_up = []
# theta_low = []
# time = []
# for i in range(len(ref)):
#     t = i*0.01
#     dx,dy,dtheta = Df(t)
#     x_up.append(ref[i][0]+dx)
#     x_low.append(ref[i][0]-dx)

#     y_up.append(ref[i][1]+dy)
#     y_low.append(ref[i][1]-dy)

#     theta_up.append(ref[i][2]+dtheta)
#     theta_low.append(ref[i][2]-dtheta)

#     time.append(t)

# plt.figure(2)
# plt.plot(time,x_up,'r')
# plt.plot(time,x_low,'r')

# plt.figure(3)
# plt.plot(time,y_up,'r')
# plt.plot(time,y_low,'r')

# plt.figure(4)
# plt.plot(time,theta_up,'r')
# plt.plot(time,theta_low,'r')

# plt.figure(1)
# plt.plot(0,0,'y.')
# plt.plot([-16,-14,-14,-16,-16],[-1,-1,1,1,-1])
plt.show()
