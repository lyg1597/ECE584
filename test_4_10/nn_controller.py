import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, 1)
        self.control = torch.nn.Linear(D_in,2)

    def forward(self,x):
        h1 = torch.nn.functional.relu(self.linear1(x))
        # h2 = torch.nn.functional.relu(self.linear2(h1))
        y = self.linear2(h1)
        u = self.control(x)
        return y,u

Lr = 2
Lf = 2
dt = 0.01

def func1(t,vars,args):
    curr_x = vars[0]
    curr_y = vars[1] 
    curr_theta = vars[2]%(np.pi*2)
    # ip = getIp(curr_x, curr_y, curr_theta)
    vr = args[0]
    delta = args[1]%(np.pi*2)

    if vr > 40:
        vr = 40
    elif vr < -40:
        vr = -40

    if delta > np.pi/4 and delta <= np.pi:
        delta = np.pi/4
    elif delta > np.pi and delta < np.pi*7/4:
        delta = np.pi*7/4

    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    dx = vr*np.cos(curr_theta+beta)
    dy = vr*np.sin(curr_theta+beta)
    dtheta = vr/Lr * np.sin(beta)
    return [dx,dy,dtheta]

pos_ref = np.arange(-15,0.01,0.1)
ref = []
eref = []
for i in range(pos_ref.shape[0]):
    ref.append([pos_ref[i],0,0])
    eref.append([0,0,i*0.01])


device = torch.device('cuda')
# data = torch.FloatTensor(input_data)
eref_tensor = torch.FloatTensor(eref)
# label = label.to(device)
model = TwoLayerNet(2,100)
# criterion = torch.nn.MSELoss(reduction='sum')
# criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)
for i in range(0,1000):
    # x_init = np.random.uniform(-16,-14)
    # y_init = np.random.uniform(-1,1)
    # theta_init = np.random.uniform(-180,180)
    x_init = -15
    y_init = 0
    theta_init = 0
    trajectory = []
    data = []
    e_tensor = torch.zeros(0,1)
    beta = torch.zeros(0,1)

    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init*np.pi/180])
    
    x = x_init
    y = y_init
    theta = theta_init
    trajectory.append([0,x_init,y_init,theta_init])
    data.append([x,y])
    for j in range(0,len(ref)):
        e_pos = np.sqrt((ref[j][0]-x)**2+(ref[j][1]-y)**2)
        e_theta = (theta - np.arctan2(ref[j][1]-y,ref[j][0]-x))%(np.pi*2)
        input_tensor = torch.FloatTensor([[e_pos,e_theta]])
        beta_tensor,u_tensor = model(input_tensor)
        beta = torch.cat((beta,beta_tensor),0)

        # vr = u_tensor.data.tolist()[0][0]
        # delta = u_tensor.data.tolist()[0][1]
        # r.set_f_params([vr,delta])
        # val = r.integrate(r.t+dt)
        # trajectory.append([r.t,val[0],val[1],val[2]])
        vr = torch.clamp(u_tensor[0][0],-100,100)
        delta = u_tensor[0][1]
        delta = delta%(np.pi*2)-np.pi
        delta = torch.clamp(delta,-np.pi/4,np.pi/4)
        delta = torch.atan(Lr/(Lr+Lf) * torch.sin(delta)/torch.cos(delta))
        new_x = x+0.01*vr*torch.cos(theta+delta)
        new_y = y+0.01*vr*torch.sin(theta+delta)
        new_theta = theta+0.01*vr/Lr*torch.sin(delta)
        
        if j+1<len(ref):
            e_pos_next = torch.sqrt((new_x-ref[j+1][0])**2+(new_y-ref[j+1][1])**2)
            e_pos_next = e_pos_next.reshape((1,1))
            e_tensor = torch.cat((e_tensor,e_pos_next))

        x = new_x.data
        y = new_y.data
        theta = new_theta.data%(np.pi*2)

    e_pos_end = torch.sqrt((new_x-0)**2+(new_y-0)**2)
    # beta0,tmp = model(eref_tensor)
    # data = torch.FloatTensor(data)
    # data = data.to(device)
    # loss = criterion(data,label)
    # loss = torch.nn.functional.relu(-beta).mean()+beta0.pow(2).mean()+\
    #     torch.nn.functional.relu(e_tensor-beta).mean()+e_tensor.mean()
    # loss = e_pos_end
    loss = e_tensor.mean()
    # if loss < 0.001:
    #     loss = beta.sum()
    
    print(i,loss.item(),e_pos_end.item(),new_x.item(),new_y.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

x_init = -15
y_init = 0
theta_init = 0
trajectory = []
r = ode(func1)
r.set_initial_value([x_init,y_init,theta_init*np.pi/180])

x = x_init
y = y_init
theta = theta_init
trajectory.append([0,x_init,y_init,theta_init])
for j in range(0,len(ref)):
    e_pos = np.sqrt((ref[j][0]-x)**2+(ref[j][1]-y)**2)
    e_theta = (theta - np.arctan2(ref[j][1]-y,ref[j][0]-x))%(np.pi*2)
    input_tensor = torch.FloatTensor([[e_pos,e_theta]])
    beta_tensor,u_tensor = model(input_tensor)
    beta = torch.cat((beta,beta_tensor),0)

    vr = u_tensor.data.tolist()[0][0]
    delta = u_tensor.data.tolist()[0][1]
    delta = delta%(np.pi*2)-np.pi

    if vr > 40:
        vr = 40
    elif vr < -40:
        vr = -40

    if delta > np.pi/4:
        delta = np.pi/4
    elif delta < -np.pi/4:
        delta = -np.pi/4

    delta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
    x = x+0.01*vr*np.cos(theta+delta)
    y = y+0.01*vr*np.sin(theta+delta)
    theta = theta+0.01*vr/Lr * np.sin(delta)
    # vr = 25
    # delta = 0
    # r.set_f_params([vr,delta])
    # val = r.integrate(r.t+dt)
    trajectory.append([r.t,x,y,theta])
    # data.append([r.t,val[0],val[1],val[2]])
    # x = val[0]
    # y = val[1]
    # theta = val[2]

t_plt = []
x_plt = []
y_plt = []
theta_plt = []
for i in range(len(trajectory)):
    t_plt.append(trajectory[i][0])
    x_plt.append(trajectory[i][1])
    y_plt.append(trajectory[i][2])

plt.plot(x_plt,y_plt,'.')
plt.show()