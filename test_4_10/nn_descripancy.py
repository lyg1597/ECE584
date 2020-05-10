import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt

class ControllerNet(torch.nn.Module):
    def __init__(self,D_in,H1):
        super(ControllerNet, self).__init__()
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

class DescripancyNet(torch.nn.Module):
    def __init__(self,D_in):
        super(DescripancyNet, self).__init__()
        # self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, 1)
        self.control1 = torch.nn.Linear(D_in,6)

    def forward(self,x):
        # h1 = torch.nn.functional.relu(self.linear1(x))
        # # h2 = torch.nn.functional.relu(self.linear2(h1))
        # y = self.linear2(h1)

        d = self.control1(x)
        return d

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

pos_ref = np.arange(-15,0.01,0.05)
ref = []
for i in range(pos_ref.shape[0]):
    ref.append([pos_ref[i],0,0])

model = ControllerNet(4,100)
model.load_state_dict(torch.load('./model_controller'))

trajectory = []
x = []
y = []
theta = []
ref_x = []
ref_y = []
ref_theta = []
time = []
for j in range(100):
    x_init = np.random.uniform(-16,-14)
    y_init = np.random.uniform(-1,1)
    theta_init = np.random.uniform(-np.pi/2,np.pi/2)
    print(j,x_init,y_init,theta_init*180/np.pi)

    trajectory.append([0,x_init,y_init,theta_init,-15,0,0])
    x.append(x_init)
    y.append(y_init)
    theta.append(theta_init)
    ref_x.append(-15)
    ref_y.append(0)
    ref_theta.append(0)
    time.append(0)

    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])

    tmp_x = [x_init]
    tmp_y = [y_init]
    tmp_traj = [[0,x_init,y_init,theta_init]]

    for i in range(len(ref)-1):
        error_x = ref[i+1][0]-tmp_traj[i][1]
        error_y = ref[i+1][1]-tmp_traj[i][2]
        error_theta = (ref[i+1][2]-tmp_traj[i][3])%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sin(error_theta)
        
        data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
        u = model(data)
        vr = u[0].item()
        delta = u[1].item()

        r.set_f_params([vr,delta])
        val = r.integrate(r.t+0.01)

        trajectory.append([r.t,val[0],val[1],val[2],ref[i+1][0],ref[i+1][1],ref[i+1][2]])
        x.append(val[0])
        y.append(val[1])
        theta.append(val[2])
        ref_x.append(ref[i+1][0])
        ref_y.append(ref[i+1][1])
        ref_theta.append(ref[i+1][2])
        time.append(r.t)

        tmp_x.append(val[0])
        tmp_y.append(val[1])
        tmp_traj.append([r.t,val[0],val[1],val[2]])
    
    plt.plot(tmp_x,tmp_y,'b')
    plt.plot(tmp_x,tmp_y,'g.')
plt.show()

x_tensor = torch.FloatTensor(x)
y_tensor = torch.FloatTensor(y)
theta_tensor = torch.FloatTensor(theta)

ref_x_tensor = torch.FloatTensor(ref_x)
ref_y_tensor = torch.FloatTensor(ref_y)
ref_theta_tensor = torch.FloatTensor(ref_theta)

time_tensor = torch.FloatTensor(time)

descripancy = DescripancyNet(3)
data = torch.FloatTensor([1,1,np.pi/2])
optimizer = torch.optim.Adam(descripancy.parameters(), lr=1e-1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.995)
for i in range(3000):
    df = descripancy(data)
    kx = df[0]
    lx = df[1]
    ky = df[2]
    ly = df[3]
    ktheta = df[4]
    ltheta = df[5]

    beta_x = kx*torch.exp(lx*time_tensor)
    beta_y = ky*torch.exp(ly*time_tensor)
    beta_theta = ktheta * np.pi/2 * torch.exp(ltheta*time_tensor)

    error_x = torch.relu(torch.abs(x_tensor - ref_x_tensor)-beta_x).mean()
    error_y = torch.relu(torch.abs(y_tensor - ref_y_tensor)-beta_y).mean()
    error_theta = torch.relu(torch.abs(theta_tensor - ref_theta_tensor)-beta_theta).mean()

    error_lx = torch.relu(lx)
    error_ly = torch.relu(ly)
    error_ltheta = torch.relu(ltheta)

    loss_constraint = error_x+error_y+error_theta+error_lx+error_ly+error_ltheta
    loss_minimize = beta_x.mean()+beta_y.mean()+beta_theta.mean()
    loss = loss_constraint*10000+loss_minimize
    if i%10 == 0:
        print(i,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

df = descripancy(data)
kx = df[0].data
lx = df[1].data
ky = df[2].data
ly = df[3].data
ktheta = df[4].data
ltheta = df[5].data

x_upper = []
x_lower = []
y_upper = []
y_lower = []
theta_upper = []
theta_lower = []
time_desc = []
for i in range(len(ref)):
    t = i*0.01
    time_desc.append(t)
    
    x_upper.append(ref[i][0]+kx*np.exp(lx*t))
    x_lower.append(ref[i][0]-kx*np.exp(lx*t))

    y_upper.append(ref[i][1]+ky*np.exp(ly*t))
    y_lower.append(ref[i][1]-ky*np.exp(ly*t))

    theta_upper.append(ref[i][1]+ktheta*np.exp(ltheta*t))
    theta_lower.append(ref[i][1]-ktheta*np.exp(ltheta*t))
    
plt.plot(time_desc,x_upper,'r')
plt.plot(time_desc,x_lower,'r')
plt.plot(time,x,'b.')
plt.show()

plt.plot(time_desc,y_upper,'r')
plt.plot(time_desc,y_lower,'r')
plt.plot(time,y,'b.')
plt.show()

plt.plot(time_desc,theta_upper,'r')
plt.plot(time_desc,theta_lower,'r')
plt.plot(time,theta,'b.')
plt.show()

# error_x_counter_tensor = torch.zeros((0,1))
# error_y_counter_tensor = torch.zeros((0,1))
# error_cos_counter_tensor = torch.zeros((0,1))
# error_sin_counter_tensor = torch.zeros((0,1))

# x_counter_tensor = torch.zeros((0,1))
# y_counter_tensor = torch.zeros((0,1))
# theta_counter_tensor = torch.zeros((0,1))

# ref_x_counter_tensor = torch.zeros((0,1))
# ref_y_counter_tensor = torch.zeros((0,1))
# ref_theta_counter_tensor = torch.zeros((0,1))

# optimizer_controller = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
# scheduler_controller = torch.optim.lr_scheduler.StepLR(optimizer_controller, step_size=50, gamma=0.995)


# x_counter = []
# y_counter = []
# theta_counter = []
# ref_x_counter = []
# ref_y_counter = []
# ref_theta_counter = []
# error_x_counter = []
# error_y_counter = []
# error_theta_counter = []
# error_cos_counter = []
# error_sin_counter = []

# for j in range(10):
#     x_init = np.random.uniform(-16,-14)
#     y_init = np.random.uniform(-1,1)
#     theta_init = np.random.uniform(-np.pi/2,np.pi/2)
#     print(j,x_init,y_init,theta_init*180/np.pi)

#     r = ode(func1)
#     r.set_initial_value([x_init,y_init,theta_init])

#     tmp_x = []
#     tmp_y = []
#     tmp_theta = []
#     tmp_traj = [[0,x_init,y_init,theta_init]]
#     tmp_ref_x = []
#     tmp_ref_y = []
#     tmp_ref_theta = []

#     error_x_counter_tmp = []
#     error_y_counter_tmp = []
#     error_theta_counter_tmp = []
#     error_cos_counter_tmp = []
#     error_sin_counter_tmp = []

#     counter_flag = False
#     for i in range(len(ref)-1):
#         tmp_x.append([tmp_traj[i][1]])
#         tmp_y.append([tmp_traj[i][2]])
#         tmp_theta.append([tmp_traj[i][3]])

#         dx = kx*np.exp(lx*t)
#         dy = ky*np.exp(ly*t)
#         dtheta = ktheta*np.exp(ltheta*t)
        
#         error_x = ref[i+1][0]-tmp_traj[i][1]
#         error_y = ref[i+1][1]-tmp_traj[i][2]
#         error_theta = (ref[i+1][2]-tmp_traj[i][3])%(np.pi*2)
#         error_theta_cos = np.cos(error_theta)
#         error_theta_sin = np.sin(error_theta)
        
#         error_x_counter_tmp.append([error_x])
#         error_y_counter_tmp.append([error_y])
#         error_theta_counter_tmp.append([error_theta])
#         error_cos_counter_tmp.append([error_theta_cos])
#         error_sin_counter_tmp.append([error_theta_sin])

#         data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
#         u = model(data)
#         vr = u[0].item()
#         delta = u[1].item()

#         r.set_f_params([vr,delta])
#         val = r.integrate(r.t+0.01)        

#         tmp_traj.append([r.t,val[0],val[1],val[2]])
#         tmp_ref_x.append([ref[i][0]])
#         tmp_ref_y.append([ref[i][1]])
#         tmp_ref_theta.append([ref[i][2]])

#         if np.abs(val[0] - ref[i+1][0]) > dx or np.abs(val[1] - ref[i+1][1]) > dy or np.abs(val[2] - ref[i+1][2]) > dtheta:
#             counter_flag = True
    

#     if counter_flag:
#         x_counter = x_counter+tmp_x
#         y_counter = y_counter+tmp_y
#         theta_counter = theta_counter+tmp_theta

#         ref_x_counter = ref_x_counter + tmp_ref_x
#         ref_y_counter = ref_y_counter + tmp_ref_y
#         ref_theta_counter = ref_theta_counter + tmp_ref_theta

#         error_x_counter = error_x_counter + error_x_counter_tmp
#         error_y_counter = error_y_counter + error_y_counter_tmp
#         error_theta_counter = error_theta_counter + error_theta_counter_tmp
#         error_cos_counter = error_cos_counter + error_cos_counter_tmp
#         error_sin_counter = error_sin_counter + error_sin_counter_tmp


# error_x_counter_tensor = torch.cat((error_x_counter_tensor,torch.FloatTensor(error_x_counter)),dim = 0)
# error_y_counter_tensor = torch.cat((error_y_counter_tensor,torch.FloatTensor(error_y_counter)),dim = 0)
# error_cos_counter_tensor = torch.cat((error_cos_counter_tensor,torch.FloatTensor(error_cos_counter)),dim = 0)
# error_sin_counter_tensor = torch.cat((error_sin_counter_tensor,torch.FloatTensor(error_sin_counter)),dim = 0)

# x_counter_tensor = torch.cat((x_counter_tensor,torch.FloatTensor(x_counter)),dim = 0)
# y_counter_tensor = torch.cat((y_counter_tensor,torch.FloatTensor(y_counter)),dim = 0)
# theta_counter_tensor = torch.cat((theta_counter_tensor,torch.FloatTensor(theta_counter)),dim = 0)
        
# ref_x_counter_tensor = torch.cat((ref_x_counter_tensor,torch.FloatTensor(ref_x_counter)),dim = 0)
# ref_y_counter_tensor = torch.cat((ref_y_counter_tensor,torch.FloatTensor(ref_y_counter)),dim = 0)
# ref_theta_counter_tensor = torch.cat((ref_theta_counter_tensor,torch.FloatTensor(ref_theta_counter)),dim = 0)

# data = torch.cat((error_x_counter_tensor,error_y_counter_tensor,error_cos_counter_tensor,error_sin_counter_tensor),dim = 1)        
# for i in range(100):
#         control_tensor = model(data)
#         vr = torch.clamp(control_tensor[:,0],-0,30)
#         delta = torch.clamp(control_tensor[:,1],-np.pi/4,np.pi/4)

#         delta = torch.atan(Lr/(Lr+Lf)*torch.sin(delta)/torch.cos(delta))
#         new_x_tensor = x_counter_tensor+0.01*vr*torch.cos(theta_counter_tensor+delta)
#         new_y_tensor = y_counter_tensor+0.01*vr*torch.sin(theta_counter_tensor+delta)
#         new_theta_tensor = theta_counter_tensor+0.01*vr/Lr*torch.sin(delta)

#         error_pos = (new_x_tensor-ref_x_counter_tensor)**2+(new_y_tensor-ref_y_counter_tensor)**2
#         error_theta = (torch.sin(new_theta_tensor-ref_theta_counter_tensor))**2

#         error = error_pos+error_theta*1.0
#         loss = error.mean()
#         if i%10 == 0:
#             print(i,loss.item(),error_pos.mean().item(),error_theta.mean().item())
#         optimizer_controller.zero_grad()
#         loss.backward()
#         optimizer_controller.step()
#         scheduler_controller.step()

# trajectory = []
# x = []
# y = []
# theta = []
# ref_x = []
# ref_y = []
# ref_theta = []
# time = []

# for j in range(10):
#     x_init = np.random.uniform(-16,-14)
#     y_init = np.random.uniform(-1,1)
#     theta_init = np.random.uniform(-np.pi/2,np.pi/2)
#     print(j,x_init,y_init,theta_init*180/np.pi)

#     trajectory.append([0,x_init,y_init,theta_init,-15,0,0])
#     x.append(x_init)
#     y.append(y_init)
#     theta.append(theta_init)
#     ref_x.append(-15)
#     ref_y.append(0)
#     ref_theta.append(0)
#     time.append(0)

#     r = ode(func1)
#     r.set_initial_value([x_init,y_init,theta_init])

#     tmp_x = [x_init]
#     tmp_y = [y_init]
#     tmp_traj = [[0,x_init,y_init,theta_init]]

#     for i in range(len(ref)-1):
#         error_x = ref[i+1][0]-tmp_traj[i][1]
#         error_y = ref[i+1][1]-tmp_traj[i][2]
#         error_theta = (ref[i+1][2]-tmp_traj[i][3])%(np.pi*2)
#         error_theta_cos = np.cos(error_theta)
#         error_theta_sin = np.sin(error_theta)
        
#         data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
#         u = model(data)
#         vr = u[0].item()
#         delta = u[1].item()

#         r.set_f_params([vr,delta])
#         val = r.integrate(r.t+0.01)

#         trajectory.append([r.t,val[0],val[1],val[2],ref[i+1][0],ref[i+1][1],ref[i+1][2]])
#         x.append(val[0])
#         y.append(val[1])
#         theta.append(val[2])
#         ref_x.append(ref[i+1][0])
#         ref_y.append(ref[i+1][1])
#         ref_theta.append(ref[i+1][2])
#         time.append(r.t)

#         tmp_x.append(val[0])
#         tmp_y.append(val[1])
#         tmp_traj.append([r.t,val[0],val[1],val[2]])
    
#     plt.plot(tmp_x,tmp_y,'b')
#     plt.plot(tmp_x,tmp_y,'g.')
# plt.show()