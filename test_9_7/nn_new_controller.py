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
n_sample = 40000
n_counter = 50

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
    # dx = 2*np.exp(-0.8*t)
    # dy = 3*np.exp(-0.5*t)
    dx = 3
    dy = 2
    dtheta = np.pi/2
    return dx,dy,dtheta

v_ref = 0.5
pos_ref = np.arange(-15,0.01,v_ref)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([0,pos_ref[i],np.pi/2])

sample_x = []
sample_y = []
sample_theta = []
sample_list = []
input_data = []
ref_x = []
ref_y = []
ref_theta = []

sample_x_origin = []
sample_y_origin = []
sample_theta_origin = []
input_data_origin = []
ref_x_origin = []
ref_y_origin = []
ref_theta_origin = []
for i in range(30):
    for j in range(n_sample):
        # if j < 1000:
        #     x = ref[i][0]
        #     y = ref[i][1]
        #     theta = ref[i][2]        
        # else:
        t = i*0.01
        dx,dy,dtheta = Df(t)
        x_origin = np.random.uniform(0-dx,0+dx)
        y_origin = np.random.uniform(-0.5-dy,-0.5+dy)
        theta_origin = np.random.uniform(np.pi/2-dtheta,np.pi/2+dtheta)

        ref_x_val = 0
        ref_y_val = 0
        ref_theta_val = np.pi/2

        sample_x_origin.append(x_origin)
        sample_y_origin.append(y_origin)
        sample_theta_origin.append(theta_origin)

        ref_x_origin.append(ref_x_val)
        ref_y_origin.append(ref_y_val)
        ref_theta_origin.append(ref_theta_val)

        next_x_ref_origin = ref_x_val
        next_y_ref_origin = ref_y_val
        next_theta_ref_origin = ref_theta_val

        error_x_origin = next_x_ref_origin - x_origin
        error_y_origin = next_y_ref_origin - y_origin
        error_theta_origin = (next_theta_ref_origin - theta_origin)%(np.pi*2)
        error_theta_cos_origin = np.cos(error_theta_origin)
        error_theta_sin_origin = np.sin(error_theta_origin)
        input_data_origin.append([error_x_origin,error_y_origin,error_theta_cos_origin,error_theta_sin_origin])
        
        ref_transform = np.random.uniform(-np.pi,np.pi)
        
        tmp1 = np.cos(ref_transform)*x_origin + np.sin(ref_transform)*y_origin
        tmp2 = - np.sin(ref_transform)*x_origin + np.cos(ref_transform)*y_origin
        x = tmp1
        y = tmp2
        theta = theta_origin - ref_transform

        sample_x.append(x)
        sample_y.append(y)
        sample_theta.append(theta)
        sample_list.append([x,y,theta])

        tmp1 = np.cos(ref_transform)*ref_x_val + np.sin(ref_transform)*ref_y_val
        tmp2 = - np.sin(ref_transform)*ref_x_val + np.cos(ref_transform)*ref_y_val
        ref_x_val = tmp1
        ref_y_val = tmp2
        ref_theta_val = ref_theta_val - ref_transform

        ref_x.append(ref_x_val)
        ref_y.append(ref_y_val)
        ref_theta.append(ref_theta_val)
        
        next_x_ref = ref_x_val
        next_y_ref = ref_y_val
        next_theta_ref = ref_theta_val

        error_x = next_x_ref - x
        error_y = next_y_ref - y
        error_theta = (next_theta_ref - theta)%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sin(error_theta)
        input_data.append([error_x,error_y,error_theta_cos,error_theta_sin])

device = torch.device('cuda')
model = TwoLayerNet(4,100)
model = model.to(device)

x_tensor = torch.FloatTensor(sample_x)
x_tensor = x_tensor.to(device)
y_tensor = torch.FloatTensor(sample_y)
y_tensor = y_tensor.to(device)
theta_tensor = torch.FloatTensor(sample_theta)
theta_tensor = theta_tensor.to(device)

ref_x_tensor = torch.FloatTensor(ref_x)
ref_x_tensor = ref_x_tensor.to(device)
ref_y_tensor = torch.FloatTensor(ref_y)
ref_y_tensor = ref_y_tensor.to(device)
ref_theta_tensor = torch.FloatTensor(ref_theta)
ref_theta_tensor = ref_theta_tensor.to(device)

data = torch.FloatTensor(input_data)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)

epoch_list = []
loss_list = []

for j in range(0,1):
    for i in range(0,10000):
        control_tensor = model(data)
        vr = torch.clamp(control_tensor[:,0],0,100)
        delta = torch.clamp(control_tensor[:,1],-np.pi/3,np.pi/3)

        delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
        new_x_tensor = x_tensor+0.01*vr*torch.cos(theta_tensor+delta)
        new_y_tensor = y_tensor+0.01*vr*torch.sin(theta_tensor+delta)
        new_theta_tensor = theta_tensor+0.01*vr/Lr*torch.sin(delta)

        error_pos = 2*(new_x_tensor-ref_x_tensor)**2+(new_y_tensor-ref_y_tensor)**2
        error_goal = new_x_tensor**2+new_y_tensor**2
        # error_theta = torch.abs(torch.sin(new_theta_tensor) - torch.sin(torch.atan2(ref_y_tensor-new_y_tensor,ref_x_tensor-new_x_tensor)))
        # error_theta = new_theta_tensor%(np.pi*2) - torch.atan2(ref_y_tensor-new_y_tensor,ref_x_tensor-new_x_tensor)%(np.pi*2)
        # error_theta = (torch.sin(new_theta_tensor) - torch.sin(ref_theta_tensor))**2
        error_theta = (torch.sin(new_theta_tensor-ref_theta_tensor))**2
        error_over = (torch.sign(new_x_tensor-ref_x_tensor)*torch.sign(x_tensor-ref_x_tensor)-1)**2
        error_constraint = torch.relu(torch.sign(-data[:,1]*delta))

        error = error_pos+error_theta*5
        # error = error_theta*10
        # error_x = torch.abs(ref_x_tensor - new_x_tensor)
        # error_y = torch.abs(ref_y_tensor - new_y_tensor)
        # error_diff = torch.abs(error_x-error_y)
        # error = error_x+error_y+error_diff
        loss = error.mean()
        if i%10 == 0:
            print(i,loss.item(),error_pos.mean().item(),error_theta.mean().item(),error_constraint.mean().item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_list.append(i)
        loss_list.append(loss.item())

#############################################################
control_tensor = model(data)
vr = torch.clamp(control_tensor[:,0],0,100)
delta = torch.clamp(control_tensor[:,1],-np.pi/3,np.pi/3)

delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
new_x_tensor = x_tensor+0.01*vr*torch.cos(theta_tensor+delta)
new_y_tensor = y_tensor+0.01*vr*torch.sin(theta_tensor+delta)
new_theta_tensor = theta_tensor+0.01*vr/Lr*torch.sin(delta)

error_pos = (new_x_tensor-ref_x_tensor)**2+(new_y_tensor-ref_y_tensor)**2

error = error_pos
loss = error.mean()

#############################################################
x_tensor_origin = torch.FloatTensor(sample_x_origin)
x_tensor_origin = x_tensor_origin.to(device)
y_tensor_origin = torch.FloatTensor(sample_y_origin)
y_tensor_origin = y_tensor_origin.to(device)
theta_tensor_origin = torch.FloatTensor(sample_theta_origin)
theta_tensor_origin = theta_tensor_origin.to(device)

ref_x_tensor_origin = torch.FloatTensor(ref_x_origin)
ref_x_tensor_origin = ref_x_tensor_origin.to(device)
ref_y_tensor_origin = torch.FloatTensor(ref_y_origin)
ref_y_tensor_origin = ref_y_tensor_origin.to(device)
ref_theta_tensor_origin = torch.FloatTensor(ref_theta_origin)
ref_theta_tensor_origin = ref_theta_tensor_origin.to(device)

data_origin = torch.FloatTensor(input_data_origin)
data_origin = data_origin.to(device)

control_tensor = model(data_origin)
vr = torch.clamp(control_tensor[:,0],0,100)
delta = torch.clamp(control_tensor[:,1],-np.pi/3,np.pi/3)

delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
new_x_tensor_origin = x_tensor_origin+0.01*vr*torch.cos(theta_tensor_origin+delta)
new_y_tensor_origin = y_tensor_origin+0.01*vr*torch.sin(theta_tensor_origin+delta)
new_theta_tensor_origin = theta_tensor_origin+0.01*vr/Lr*torch.sin(delta)

error_pos = (new_x_tensor_origin-ref_x_tensor_origin)**2+(new_y_tensor_origin-ref_y_tensor_origin)**2

error = error_pos
loss_origin = error.mean()

print("without transformation", loss.item(),"with transformation", loss_origin.item())


print(data.shape)
device = torch.device('cpu')
model = model.to(device)
# x_init = np.random.uniform(-1,1)
# y_init = np.random.uniform(-16,-14)
x_init = 0
y_init = -15
theta_init = np.pi/2

trajectory = [[0,x_init,y_init,theta_init]]
r = ode(func1)
r.set_initial_value([x_init,y_init,theta_init])

for i in range(len(ref)-1):
    error_x = ref[i+1][0]-trajectory[i][1]
    error_y = ref[i+1][1]-trajectory[i][2]
    error_theta = (ref[i+1][2]-trajectory[i][3])%(np.pi*2)
    error_theta_cos = np.cos(error_theta)
    error_theta_sin = np.sin(error_theta)

    data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
    u = model(data)
    vr = u[0].item()
    delta = u[1].item()

    r.set_f_params([vr,delta])
    val = r.integrate(r.t+0.01)

    trajectory.append([r.t,val[0],val[1],val[2]])
    error_pos = np.sqrt((val[0]-ref[i+1][0])**2+(val[1]-ref[i+1][1])**2)
    
    # print(i,vr,delta,error_pos,error_x,error_y,error_theta)

x = []
y = []
for i in range(len(trajectory)):
    x.append(trajectory[i][1])
    y.append(trajectory[i][2])

plt.plot(x,y)
plt.plot(x,y,'.')

plt.show()

torch.save(model.state_dict(), './model_controller_no_sym')


new_x_tensor = new_x_tensor.to(device)
x_end = new_x_tensor.tolist()
new_y_tensor = new_y_tensor.to(device)
y_end = new_y_tensor.tolist()
new_theta_tensor = new_theta_tensor.to(device)
theta_end = new_theta_tensor.tolist()

# for i in range(max(len(sample_x),100)):
#     plt.plot([sample_x[i]-ref_x[i],x_end[i]-ref_x[i]],[sample_y[i]-ref_y[i],y_end[i]-ref_y[i]],'b')
#     plt.plot(sample_x[i]-ref_x[i],sample_y[i]-ref_y[i],'g.')
#     plt.plot(x_end[i]-ref_x[i],y_end[i]-ref_y[i],'r.')

plt.plot(sample_x,sample_y,'b.')

plt.plot(0,0,'y.')
plt.show()

plt.plot(epoch_list, loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs Epoch')
plt.show()