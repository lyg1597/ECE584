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

plt.figure(1)
plt.figure(2)
plt.figure(3)

x_start = []
y_start = []
x_end = []
y_end = []

data_test = []
label = []
x_test = []
y_test = []
theta_test = []
ref_x = []
ref_y = []
ref_theta = []


x_init = np.random.uniform(-16,-14)
y_init = np.random.uniform(-1,1)
# x_init = -15
# y_init = 0
# theta_init = 0
theta_init = np.random.uniform(-np.pi/2,np.pi/2)
for i in range(1):
    # x_init = np.random.uniform(-16,-14)
    # y_init = np.random.uniform(-1,1)
    # x_init = -15
    # y_init = 0
    # theta_init = 0
    # theta_init = np.random.uniform(-np.pi/2,np.pi/2)

    trajectory = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])
    time = [0]
    for i in range(len(ref)-1):
        error_x = ref[i+1][0]-trajectory[i][1]
        error_y = ref[i+1][1]-trajectory[i][2]
        error_theta = (ref[i+1][2]-trajectory[i][3])%(np.pi*2)
        x_start.append(trajectory[i][1]-ref[i+1][0])
        y_start.append(trajectory[i][2]-ref[i+1][1])
        data_test.append([error_x,error_y,error_theta])
        x_test.append(trajectory[i][1])
        y_test.append(trajectory[i][2])
        theta_test.append(trajectory[i][3])
        
        ref_x.append(ref[i+1][0])
        ref_y.append(ref[i+1][1])
        ref_theta.append(ref[i+1][2])
        ref.append([ref[i+1][0],ref[i+1][1],ref[i+1][2]])

        data = torch.FloatTensor([error_x,error_y,error_theta])
        u = model(data)
        vr = u[0].item()
        delta = u[1].item()

        r.set_f_params([vr,delta])
        val = r.integrate(r.t+0.01)

        x_end.append(val[0]-ref[i+1][0])
        y_end.append(val[1]-ref[i+1][1])
        trajectory.append([r.t,val[0],val[1],val[2]])
        error_pos = np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
        
        time.append(r.t)
        print(i,vr,delta,error_pos,val[0],val[1],val[2]*180/np.pi,ref[i+1][0])
    print(x_init,y_init,theta_init*180/np.pi)

    x = []
    y = []
    for i in range(len(trajectory)):
        x.append(trajectory[i][1])
        y.append(trajectory[i][2])

    plt.figure(1)
    plt.plot(x,y,'b')
    plt.plot(x,y,'g.')
    plt.plot(x_init,y_init,'r.')

    plt.figure(2)
    plt.plot(time,x,'b')
    
    plt.figure(3)
    plt.plot(time,y,'b')

plt.figure(1)
plt.plot(0,0,'y.')
plt.plot([-16,-14,-14,-16,-16],[-1,-1,1,1,-1])
plt.show()

for i in range(len(x_start)):
    plt.plot([x_start[i],x_end[i]],[y_start[i],y_end[i]],'b')
    plt.plot(x_start[i],y_start[i],'.g')
    plt.plot(x_end[i],y_end[i],'.r')

plt.plot(0,0,'.y')
plt.show()

device = torch.device('cuda')
model = model.to(device)

x_tensor = torch.FloatTensor(x_test)
x_tensor = x_tensor.to(device)
y_tensor = torch.FloatTensor(y_test)
y_tensor = y_tensor.to(device)
theta_tensor = torch.FloatTensor(theta_test)
theta_tensor = theta_tensor.to(device)

ref_x_tensor = torch.FloatTensor(ref_x)
ref_x_tensor = ref_x_tensor.to(device)
ref_y_tensor = torch.FloatTensor(ref_y)
ref_y_tensor = ref_y_tensor.to(device)
ref_theta_tensor = torch.FloatTensor(ref_theta)
ref_theta_tensor = ref_theta_tensor.to(device)


data = torch.FloatTensor(data_test)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)

for j in range(0,400):
    control_tensor = model(data)
    vr = torch.clamp(control_tensor[:,0],-30,30)
    delta = torch.clamp(control_tensor[:,1],-np.pi/4,np.pi/4)

    delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
    new_x_tensor = x_tensor+0.01*vr*torch.cos(theta_tensor+delta)
    new_y_tensor = y_tensor+0.01*vr*torch.sin(theta_tensor+delta)
    new_theta_tensor = theta_tensor+0.01*vr/Lr*torch.sin(delta)
    error_pos = (new_x_tensor-ref_x_tensor)**2+(new_y_tensor-ref_y_tensor)**2
    error_theta = (torch.sin(new_theta_tensor) - torch.sin(ref_theta_tensor))**2
    error = error_pos + error_theta*0.3
    loss = error.mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    print(i,loss.item())

device = torch.device('cpu')
model = model.to(device)
trajectory = [[0,x_init,y_init,theta_init]]
r = ode(func1)
r.set_initial_value([x_init,y_init,theta_init])

for i in range(len(ref)-1):
    error_x = ref[i+1][0]-trajectory[i][1]
    error_y = ref[i+1][1]-trajectory[i][2]
    error_theta = (ref[i+1][2]-trajectory[i][3])%(np.pi*2)

    data = torch.FloatTensor([error_x,error_y,error_theta])
    u = model(data)
    vr = u[0].item()
    delta = u[1].item()

    r.set_f_params([vr,delta])
    val = r.integrate(r.t+0.01)

    trajectory.append([r.t,val[0],val[1],val[2]])
    error_pos = np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
    
    print(i,vr,delta,error_pos,error_x,error_y,error_theta)

x = []
y = []
for i in range(len(trajectory)):
    x.append(trajectory[i][1])
    y.append(trajectory[i][2])

plt.plot(x,y)
plt.plot(x,y,'.')

plt.show()