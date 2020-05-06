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
n_sample = 3000
n_counter = 50

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
    dx = 2*np.exp(-0.8*t)
    dy = 3*np.exp(-0.5*t)
    dtheta = np.pi/2
    return dx,dy,dtheta

v_ref = 0.05
pos_ref = np.arange(-15,0.01,v_ref)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([pos_ref[i],0,0])

sample_x = []
sample_y = []
sample_theta = []
sample_list = []
input_data = []
ref_x = []
ref_y = []
ref_theta = []
for i in range(len(ref)-1):
    for j in range(n_sample):
        # if j < 1000:
        #     x = ref[i][0]
        #     y = ref[i][1]
        #     theta = ref[i][2]        
        # else:
        t = i*0.01
        dx,dy,dtheta = Df(t)
        x = np.random.uniform(ref[i][0]-dx,ref[i][0]+dx)
        y = np.random.uniform(ref[i][1]-dy,ref[i][1]+dy)
        theta = np.random.uniform(ref[i][2]-dtheta,ref[i][2]+dtheta)
        
        sample_x.append(x)
        sample_y.append(y)
        sample_theta.append(theta)
        sample_list.append([x,y,theta])

        ref_x.append(ref[i+1][0])
        ref_y.append(ref[i+1][1])
        ref_theta.append(ref[i+1][2])
        
        next_x_ref = ref[i+1][0]
        next_y_ref = ref[i+1][1]
        next_theta_ref = ref[i+1][2]

        error_x = next_x_ref - x
        error_y = next_y_ref - y
        error_theta = (next_theta_ref - theta)%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sin(error_theta)
        input_data.append([error_x,error_y,error_theta_cos,error_theta_sin])
    
# for i in range(n_sample):
#     x = np.random.uniform(-0.01-3,-0.01+3)
#     y = np.random.uniform(0-3,0+3)
#     theta = np.random.uniform(0-np.pi,0+np.pi)

#     sample_x.append(x)
#     sample_y.append(y)
#     sample_theta.append(theta)
#     sample_list.append([x,y,theta])

#     ref_x.append(0)
#     ref_y.append(0)
#     ref_theta.append(0)

#     next_x_ref = 0
#     next_y_ref = 0
#     next_theta_ref = 0

#     error_x = next_x_ref - x
#     error_y = next_y_ref - y
#     error_theta = (next_theta_ref - theta)%(np.pi*2)
#     input_data.append([error_x,error_y,error_theta])


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

for j in range(0,1):
    for i in range(0,5000):
        control_tensor = model(data)
        vr = torch.clamp(control_tensor[:,0],-0,30)
        delta = torch.clamp(control_tensor[:,1],-np.pi/4,np.pi/4)

        delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
        new_x_tensor = x_tensor+0.01*vr*torch.cos(theta_tensor+delta)
        new_y_tensor = y_tensor+0.01*vr*torch.sin(theta_tensor+delta)
        new_theta_tensor = theta_tensor+0.01*vr/Lr*torch.sin(delta)

        error_pos = (new_x_tensor-ref_x_tensor)**2+(new_y_tensor-ref_y_tensor)**2
        error_goal = new_x_tensor**2+new_y_tensor**2
        # error_theta = torch.abs(torch.sin(new_theta_tensor) - torch.sin(torch.atan2(ref_y_tensor-new_y_tensor,ref_x_tensor-new_x_tensor)))
        # error_theta = new_theta_tensor%(np.pi*2) - torch.atan2(ref_y_tensor-new_y_tensor,ref_x_tensor-new_x_tensor)%(np.pi*2)
        # error_theta = (torch.sin(new_theta_tensor) - torch.sin(ref_theta_tensor))**2
        error_theta = (torch.sin(new_theta_tensor-ref_theta_tensor))**2
        error_over = (torch.sign(new_x_tensor-ref_x_tensor)*torch.sign(x_tensor-ref_x_tensor)-1)**2

        error = error_pos+error_theta*1.0
        # error = error_theta*10
        # error_x = torch.abs(ref_x_tensor - new_x_tensor)
        # error_y = torch.abs(ref_y_tensor - new_y_tensor)
        # error_diff = torch.abs(error_x-error_y)
        # error = error_x+error_y+error_diff
        loss = error.mean()
        if i%10 == 0:
            print(i,loss.item(),error_pos.mean().item(),error_theta.mean().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # x_counter = torch.zeros((0,1),device = device)
    # y_counter = torch.zeros((0,1),device = device)
    # theta_counter = torch.zeros((0,1),device = device)
    
    # x_ref_counter = torch.zeros((0,1),device = device)
    # y_ref_counter = torch.zeros((0,1),device = device)
    # theta_ref_counter = torch.zeros((0,1),device = device)

    # error_x_counter = torch.zeros((0,1),device = device)
    # error_y_counter = torch.zeros((0,1),device = device)
    # error_theta_counter = torch.zeros((0,1),device = device)
    # error_pos_init = torch.zeros((0,1),device = device)
    
    # for i in range(len(ref)-1):
    #     t = i*0.01
    #     dx,dy,dtheta = Df(t)
    #     x_counter_tmp = torch.rand(n_counter,1,device = device)*dx*2-dx + ref[i][0]
    #     x_counter = torch.cat((x_counter,x_counter_tmp),dim = 0)
    #     y_counter_tmp = torch.rand(n_counter,1,device = device)*dy*2-dy + ref[i][1]
    #     y_counter = torch.cat((y_counter,y_counter_tmp),dim = 0)
    #     theta_counter_tmp = torch.rand(n_counter,1,device = device)*dtheta*2-dtheta + ref[i][2]
    #     theta_counter = torch.cat((theta_counter,theta_counter_tmp),dim = 0)
        

    #     x_ref_counter_tmp = torch.ones([n_counter,1],device = device) * ref[i+1][0]
    #     x_ref_counter = torch.cat((x_ref_counter,x_ref_counter_tmp),dim = 0)
    #     y_ref_counter_tmp = torch.ones([n_counter,1],device = device) * ref[i+1][1]
    #     y_ref_counter = torch.cat((y_ref_counter,y_ref_counter_tmp),dim = 0)
    #     theta_ref_counter_tmp = torch.ones([n_counter,1],device = device) * ref[i+1][2]    
    #     theta_ref_counter = torch.cat((theta_ref_counter,theta_ref_counter_tmp),dim = 0)
        
    #     error_x_counter_tmp = x_ref_counter_tmp - x_counter_tmp
    #     error_x_counter = torch.cat((error_x_counter,error_x_counter_tmp),dim = 0)
    #     error_y_counter_tmp = y_ref_counter_tmp - y_counter_tmp
    #     error_y_counter = torch.cat((error_y_counter,error_y_counter_tmp),dim = 0)
    #     error_theta_counter_tmp = (theta_ref_counter_tmp - theta_counter_tmp)%(np.pi*2)
    #     error_theta_counter = torch.cat((error_theta_counter,error_theta_counter_tmp),dim = 0)
    #     error_pos_init_tmp = torch.sqrt(error_x_counter_tmp**2+error_y_counter_tmp**2)
    #     error_pos_init = torch.cat((error_pos_init,error_pos_init_tmp),dim = 0)

    # error_theta_cos_counter = torch.cos(error_theta_counter)
    # error_theta_sin_counter = torch.sin(error_theta_counter)

    # data_counter = torch.cat((error_x_counter,error_y_counter,error_theta_cos_counter,error_theta_sin_counter),dim = 1)
    # control_tensor = model(data_counter)
    # vr = torch.clamp(control_tensor[:,0],-0,30)
    # delta = torch.clamp(control_tensor[:,1],-np.pi/4,np.pi/4)
    # new_x_counter = x_counter[:,0]+0.01*vr*torch.cos(theta_counter[:,0]+delta)
    # new_y_counter = y_counter[:,0]+0.01*vr*torch.sin(theta_counter[:,0]+delta)
    # new_theta_counter = theta_counter[:,0]+0.01*vr/Lr*torch.sin(delta)
    # error_x_end = x_ref_counter[:,0] - new_x_counter
    # error_y_end = y_ref_counter[:,0] - new_y_counter
    # error_pos_end = torch.sqrt(error_x_end**2+error_y_end**2)
    # l = 0
    # for i in range(error_pos_init.shape[0]):
    #     if error_pos_init[i].item()<error_pos_end[i].item():
    #         l += 1
    #         data = torch.cat((data,data_counter[i:i+1]),dim = 0)
    #         x_tensor = torch.cat((x_tensor,x_counter[i]),dim = 0)
    #         y_tensor = torch.cat((y_tensor,y_counter[i]),dim = 0)
    #         theta_tensor = torch.cat((theta_tensor,theta_counter[i]),dim = 0)
    #         ref_x_tensor = torch.cat((ref_x_tensor,x_ref_counter[i]),dim = 0)
    #         ref_y_tensor = torch.cat((ref_y_tensor,y_ref_counter[i]),dim = 0)
    #         ref_theta_tensor = torch.cat((ref_theta_tensor,theta_ref_counter[i]),dim = 0)

    # print(j,l)

print(data.shape)
device = torch.device('cpu')
model = model.to(device)
x_init = np.random.uniform(-16,-15)
y_init = np.random.uniform(-1,1)
# x_init = -15
# y_init = 0
theta_init = np.random.uniform(-np.pi,np.pi)

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
    
    print(i,vr,delta,error_pos,error_x,error_y,error_theta)

x = []
y = []
for i in range(len(trajectory)):
    x.append(trajectory[i][1])
    y.append(trajectory[i][2])

plt.plot(x,y)
plt.plot(x,y,'.')

plt.show()

torch.save(model.state_dict(), './model_controller')


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