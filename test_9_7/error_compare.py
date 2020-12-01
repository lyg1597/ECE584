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

def controller(model, state, ref):
    x_transform = 0 - ref[1][0]
    y_transform = 0 - ref[1][1]
    theta_transform = np.arctan2(ref[1][1]-ref[0][1],ref[1][0]-ref[0][0]) - np.pi/2

    x = state[1]
    y = state[2]
    theta = state[3]

    ref_x = ref[1][0]
    ref_y = ref[1][1]
    ref_theta = ref[1][2]

    x = x + x_transform
    y = y + y_transform
    tmp1 = np.cos(theta_transform)*x + np.sin(theta_transform)*y
    tmp2 = -np.sin(theta_transform)*x + np.cos(theta_transform)*y
    x = tmp1
    y = tmp2
    theta = theta - theta_transform

    ref_x = ref_x + x_transform
    ref_y = ref_y + y_transform
    tmp1 = np.cos(theta_transform)*ref_x + np.sin(theta_transform)*ref_y
    tmp2 = -np.sin(theta_transform)*ref_x + np.cos(theta_transform)*ref_y
    ref_x = tmp1
    ref_y = tmp2
    ref_theta = ref_theta - theta_transform    

    error_x = ref_x-x
    error_y = ref_y-y
    error_theta = (ref_theta - theta)%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    u = model(data)
    vr = u[0].item()
    delta = u[1].item()

    return vr, delta, error_x, error_y

def controller_no_sym(model, state, ref):
    x = state[1]
    y = state[2]
    theta = state[3]

    ref_x = ref[1][0]
    ref_y = ref[1][1]
    ref_theta = ref[1][2]  

    error_x = ref_x-x
    error_y = ref_y-y
    error_theta = (ref_theta - theta)%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    u = model(data)
    vr = u[0].item()
    delta = u[1].item()

    return vr, delta, error_x, error_y

pos_ref = np.arange(-5,0.01,0.5)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([0,pos_ref[i],np.pi/2])
    eref.append([0,0,i*0.01])

model = TwoLayerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

model_no_sym = TwoLayerNet(4,100)
model_no_sym.load_state_dict(torch.load('./model_controller_no_sym'))

plt.figure(1)
plt.figure(2)
plt.figure(3)

x_start = []
y_start = []
x_end = []
y_end = []
data = []
label = []
vr_list = []
delta_list = []
x_init_fix = np.random.uniform(-1,1)
y_init_fix = np.random.uniform(-6,-4)
# x_init_fix = 0
# y_init_fix = -5
theta_init_fix = np.pi/2
first_run = True
ref_fix = copy.deepcopy(ref)
error_x_list = []
error_y_list = []
x_diff = []
y_diff = []

error_symmetry_list = []
error_no_symmetry_list = []
error_list = []

# Four different runs
for i in range(8):
    if first_run:
        x_init = x_init_fix
        y_init = y_init_fix
        theta_init = theta_init_fix
        first_run = False
    else:
        theta_transform = np.pi/4*i
        
        tmp1 = np.cos(theta_transform)*x_init_fix + np.sin(theta_transform)*y_init_fix
        tmp2 = -np.sin(theta_transform)*x_init_fix + np.cos(theta_transform)*y_init_fix
        x_init = tmp1
        y_init = tmp2
        theta_init = theta_init_fix - theta_transform

        for j in range(len(ref_fix)):
            tmp1 = np.cos(theta_transform)*ref_fix[j][0] + np.sin(theta_transform)*ref_fix[j][1]
            tmp2 = -np.sin(theta_transform)*ref_fix[j][0] + np.cos(theta_transform)*ref_fix[j][1]
            ref[j][0] = tmp1
            ref[j][1] = tmp2
            ref[j][2] = ref_fix[j][2] - theta_transform

    print("initial",x_init,y_init,theta_init*180/np.pi)

    trajectory = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])
    time = [0]
    vr_list_tmp = []
    delta_list_tmp = []
    error_pos = 0
    for i in range(len(ref)-1):

        vr, delta, _, _ = controller(model, trajectory[i], ref[i:i+2])
        # vr = 20
        # delta = 0

        vr_list_tmp.append(vr)
        delta_list_tmp.append(delta)
        # delta = delta * np.sign(symmetry_y)

        if vr > 100:
            vr = 100
        elif vr < -0:
            vr = -0

        if delta > np.pi/3: 
            delta = np.pi/3
        elif delta < -np.pi/3:
            delta = -np.pi/3

        beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))

        val = [0,0,0]
        val[0] = trajectory[i][1] + 0.01*vr*np.cos(trajectory[i][3]+beta)
        val[1] = trajectory[i][2] + 0.01*vr*np.sin(trajectory[i][3]+beta)
        val[2] = trajectory[i][3] + 0.01*vr/Lr * np.sin(beta)

        # r.set_f_params([vr,delta])
        # val = r.integrate(r.t+0.01)

        x_end.append(val[0]-ref[i+1][0])
        y_end.append(val[1]-ref[i+1][1])
        # trajectory.append([r.t,np.around(val[0], decimals = 4),np.around(val[1], decimals = 5),np.around(val[2], decimals = 5)])
        trajectory.append([r.t,val[0],val[1],val[2]])
        error_pos += np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
        
        time.append(r.t)
        print(i,vr,delta,error_pos)

    error_symmetry_list.append(error_pos)
    
    trajectory2 = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])
    time = [0]
    vr_list_tmp = []
    delta_list_tmp = []
    error_pos = 0
    for i in range(len(ref)-1):

        vr, delta, _, _ = controller_no_sym(model, trajectory2[i], ref[i:i+2])
        # vr = 20
        # delta = 0

        vr_list_tmp.append(vr)
        delta_list_tmp.append(delta)
        # delta = delta * np.sign(symmetry_y)

        if vr > 100:
            vr = 100
        elif vr < -0:
            vr = -0

        if delta > np.pi/3: 
            delta = np.pi/3
        elif delta < -np.pi/3:
            delta = -np.pi/3

        beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))

        val = [0,0,0]
        val[0] = trajectory2[i][1] + 0.01*vr*np.cos(trajectory2[i][3]+beta)
        val[1] = trajectory2[i][2] + 0.01*vr*np.sin(trajectory2[i][3]+beta)
        val[2] = trajectory2[i][3] + 0.01*vr/Lr * np.sin(beta)

        # r.set_f_params([vr,delta])
        # val = r.integrate(r.t+0.01)

        x_end.append(val[0]-ref[i+1][0])
        y_end.append(val[1]-ref[i+1][1])
        # trajectory2.append([r.t,np.around(val[0], decimals = 4),np.around(val[1], decimals = 5),np.around(val[2], decimals = 5)])
        trajectory2.append([r.t,val[0],val[1],val[2]])
        error_pos += np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
        
        time.append(r.t)
        print(i,vr,delta,error_pos)

    error_no_symmetry_list.append(error_pos)

    trajectory3 = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])
    time = [0]
    vr_list_tmp = []
    delta_list_tmp = []
    error_pos = 0
    for i in range(len(ref)-1):

        vr, delta, _, _ = controller_no_sym(model_no_sym, trajectory3[i], ref[i:i+2])
        # vr = 20
        # delta = 0

        vr_list_tmp.append(vr)
        delta_list_tmp.append(delta)
        # delta = delta * np.sign(symmetry_y)

        if vr > 100:
            vr = 100
        elif vr < -0:
            vr = -0

        if delta > np.pi/3: 
            delta = np.pi/3
        elif delta < -np.pi/3:
            delta = -np.pi/3

        beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))

        val = [0,0,0]
        val[0] = trajectory3[i][1] + 0.01*vr*np.cos(trajectory3[i][3]+beta)
        val[1] = trajectory3[i][2] + 0.01*vr*np.sin(trajectory3[i][3]+beta)
        val[2] = trajectory3[i][3] + 0.01*vr/Lr * np.sin(beta)

        # r.set_f_params([vr,delta])
        # val = r.integrate(r.t+0.01)

        x_end.append(val[0]-ref[i+1][0])
        y_end.append(val[1]-ref[i+1][1])
        # trajectory3.append([r.t,np.around(val[0], decimals = 4),np.around(val[1], decimals = 5),np.around(val[2], decimals = 5)])
        trajectory3.append([r.t,val[0],val[1],val[2]])
        error_pos = error_pos + np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
        
        time.append(r.t)
        print(i,vr,delta,error_pos)

    error_list.append(error_pos)

    vr_list.append(vr_list_tmp)
    delta_list.append(delta_list_tmp)

    print(x_init,y_init,theta_init*180/np.pi)

    x = []
    y = []
    x_diff_tmp = []
    y_diff_tmp = []
    theta = []
    ref_x = []
    ref_y = []
    error_x_list_tmp = []
    error_y_list_tmp = []
    for i in range(len(trajectory)):
        x.append(trajectory[i][1])
        y.append(trajectory[i][2])
        theta.append(trajectory[i][3])
        ref_x.append(ref[i][0])
        ref_y.append(ref[i][1])
        error_x_list_tmp.append(trajectory[i][1] - ref[i][0])
        error_y_list_tmp.append(trajectory[i][2] - ref[i][1])
        if i>=1:
            x_diff_tmp.append(x[i]-x[i-1])
            y_diff_tmp.append(y[i]-y[i-1])
    error_x_list.append(error_x_list_tmp)
    error_y_list.append(error_y_list_tmp)
    x_diff.append(x_diff_tmp)
    y_diff.append(y_diff_tmp)

    plt.figure(1)
    plt.plot(x,y,'b')
    plt.plot(x_init,y_init,'r.')
    plt.plot(ref_x,ref_y,'g--')
    plt.plot(ref_x[0],ref_y[0],'y.')
    plt.plot(ref_x[-1],ref_y[-1],'y.')

    x2 = []
    y2 = []
    theta2 = []
    ref_x2 = []
    ref_y2 = []
    for i in range(len(trajectory2)):
        x2.append(trajectory2[i][1])
        y2.append(trajectory2[i][2])
        theta2.append(trajectory2[i][3])
        ref_x2.append(ref[i][0])
        ref_y2.append(ref[i][1])

    plt.figure(2)
    plt.plot(x2,y2,'b')
    plt.plot(x_init,y_init,'r.')
    plt.plot(ref_x2,ref_y2,'g--')
    plt.plot(ref_x2[0],ref_y2[0],'y.')
    plt.plot(ref_x2[-1],ref_y2[-1],'y.')

    x3 = []
    y3 = []
    theta3 = []
    ref_x3 = []
    ref_y3 = []
    for i in range(len(trajectory3)):
        x3.append(trajectory3[i][1])
        y3.append(trajectory3[i][2])
        theta3.append(trajectory3[i][3])
        ref_x3.append(ref[i][0])
        ref_y3.append(ref[i][1])

    plt.figure(3)
    plt.plot(x3,y3,'b')
    plt.plot(x_init,y_init,'r.')
    plt.plot(ref_x3,ref_y3,'g--')
    plt.plot(ref_x3[0],ref_y3[0],'y.')
    plt.plot(ref_x3[-1],ref_y3[-1],'y.')

print(error_symmetry_list)
print(error_no_symmetry_list)
print(error_list)

x_up = []
x_low = []
y_up = []
y_low = []
theta_up = []
theta_low = []
time = []
for i in range(len(ref)):
    t = i*0.01
    dx,dy,dtheta = Df(t)
    x_up.append(ref[i][0]+dx)
    x_low.append(ref[i][0]-dx)

    y_up.append(ref[i][1]+dy)
    y_low.append(ref[i][1]-dy)

    theta_up.append(ref[i][2]+dtheta)
    theta_low.append(ref[i][2]-dtheta)

    time.append(t)

plt.figure(2)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Vehicle Trajectory without Rotation with training symmetry')
plt.xlim((-5.5,5.5))
plt.ylim((-5.5,5.5))

plt.figure(3)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Vehicle Trajectory without rotation without training symmetry')
plt.xlim((-5.5,5.5))
plt.ylim((-5.5,5.5))

# plt.figure(1)
# plt.plot(0,0,'y.')
# plt.plot([-16,-14,-14,-16,-16],[-1,-1,1,1,-1])
plt.figure(1)
plt.xlabel('x position')
plt.ylabel('y position')
plt.title('Vehicle Trajectory without Rotation with symmetry')
plt.xlim((-5.5,5.5))
plt.ylim((-5.5,5.5))
plt.show()

# for i in range(len(x_start)):
#     plt.plot([x_start[i],x_end[i]],[y_start[i],y_end[i]],'b')
#     plt.plot(x_start[i],y_start[i],'.g')
#     plt.plot(x_end[i],y_end[i],'.r')

# plt.plot(0,0,'.y')
# plt.show()
# x = [x_init]
# y = [y_init]
# theta = [theta_init]
# r = ode(func1)
# r.set_initial_value([x_init,y_init,theta_init])
# for i in range((len(ref)-1)):
#     # if i%10 == 0:
#     #     r.set_f_params([vr_list[int(i/10)],delta_list[int(i/10)]])
#     r.set_f_params([vr_list[i],delta_list[i]])
#     val = r.integrate(r.t+0.01)
#     x.append(val[0])
#     y.append(val[1])
#     theta.append(val[2])

# plt.plot(x,y)
# plt.plot(x,y,'.r')
# plt.show()
# plt.plot(theta)
# plt.plot(theta,'.r')
# plt.show()