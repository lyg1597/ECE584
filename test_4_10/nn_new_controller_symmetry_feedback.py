import torch
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

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

def Df(t):
    dx = 2*np.exp(-0.8*t)
    dy = 3*np.exp(-0.5*t)
    dtheta = np.pi/2
    return dx,dy,dtheta

Lr = 2
Lf = 2
dt = 0.01
n_sample = 1000
n_counter = 10

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
symmetry_y = []

plt.figure(5)
plt.figure(6)
plt.figure(7)

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

        symmetry_y.append(y - next_y_ref)

        error_x = next_x_ref - x
        error_y = np.abs(next_y_ref - y)
        error_theta = (next_theta_ref - theta)%(np.pi*2)
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sign(y - next_y_ref)*np.sin(error_theta)
        input_data.append([error_x,error_y,error_theta_cos,error_theta_sin])
    
# for i in range(n_sample):
#     x = np.random.uniform(-0.01-3,-0.01+3)
#     y = np.random.uniform(0-3,0+3)
#     theta = np.random.uniform(0-np.pi/2,0+np.pi/2)

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


model = TwoLayerNet(4,100)
device = torch.device('cuda')
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

symmetry_y_tensor = torch.FloatTensor(symmetry_y)
symmetry_y_tensor = symmetry_y_tensor.to(device)

data = torch.FloatTensor(input_data)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)

descripancy = DescripancyNet(3)
optimizer_desc = torch.optim.Adam(descripancy.parameters(), lr=1e-1, weight_decay=1e-5)
scheduler_desc = torch.optim.lr_scheduler.StepLR(optimizer_desc, step_size=50, gamma=0.995)

for j in range(0,10):
    for i in range(0,2000):
        control_tensor = model(data)
        vr = torch.clamp(control_tensor[:,0],0,30)
        delta = control_tensor[:,1] * torch.sign(symmetry_y_tensor)
        # delta = control_tensor[:,1]
        delta = torch.clamp(delta,-np.pi/4,np.pi/4)

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
        error_constraint = torch.relu(torch.sign(-data[:,1]*delta))

        error = error_pos+error_theta*1.0
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

    device = torch.device('cpu')
    model = model.to(device)

    trajectory_desc = []
    x_desc = []
    y_desc = []
    theta_desc = []
    ref_x_desc = []
    ref_y_desc = []
    ref_theta_desc = []
    time_desc = []
    for k in range(100):
        x_init = np.random.uniform(-16,-14)
        y_init = np.random.uniform(-1,1)
        theta_init = np.random.uniform(-np.pi/2,np.pi/2)
        print(j,x_init,y_init,theta_init*180/np.pi)

        trajectory_desc.append([0,x_init,y_init,theta_init,-15,0,0])
        x_desc.append(x_init)
        y_desc.append(y_init)
        theta_desc.append(theta_init)
        ref_x_desc.append(-15)
        ref_y_desc.append(0)
        ref_theta_desc.append(0)
        time_desc.append(0)

        r = ode(func1)
        r.set_initial_value([x_init,y_init,theta_init])

        tmp_x = [x_init]
        tmp_y = [y_init]
        tmp_traj = [[0,x_init,y_init,theta_init]]

        for i in range(len(ref)-1):
            symmetry_y = tmp_traj[i][2]-ref[i+1][1]

            error_x = ref[i+1][0]-tmp_traj[i][1]
            error_y = np.abs(ref[i+1][1]-tmp_traj[i][2])
            error_theta = (ref[i+1][2]-tmp_traj[i][3])%(np.pi*2)
            error_theta_cos = np.cos(error_theta)
            error_theta_sin = np.sign(symmetry_y)*np.sin(error_theta)
            
            data_tmp = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
            u = model(data_tmp)
            vr = u[0].item()
            delta = u[1].item()
            delta = delta * np.sign(symmetry_y)

            val = [0,0,0]
            if vr > 30:
                vr = 30
            elif vr < -0:
                vr = -0

            if delta > np.pi/4: 
                delta = np.pi/4
            elif delta < -np.pi/4:
                delta = -np.pi/4

            beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
            val[0] = tmp_traj[i][1]+0.01*vr*np.cos(tmp_traj[i][3]+beta)
            val[1] = tmp_traj[i][2]+0.01*vr*np.sin(tmp_traj[i][3]+beta)
            val[2] = tmp_traj[i][3]+0.01*vr/Lr * np.sin(beta)

            t = (i+1)*0.01
            trajectory_desc.append([t,val[0],val[1],val[2],ref[i+1][0],ref[i+1][1],ref[i+1][2]])
            x_desc.append(val[0])
            y_desc.append(val[1])
            theta_desc.append(val[2])
            ref_x_desc.append(ref[i+1][0])
            ref_y_desc.append(ref[i+1][1])
            ref_theta_desc.append(ref[i+1][2])
            time_desc.append(t)

            tmp_x.append(val[0])
            tmp_y.append(val[1])
            tmp_traj.append([t,val[0],val[1],val[2]])
        
    #     plt.plot(tmp_x,tmp_y,'b')
    #     plt.plot(tmp_x,tmp_y,'g.')
    # plt.title('x vs y')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    x_tensor_desc = torch.FloatTensor(x_desc)
    y_tensor_desc = torch.FloatTensor(y_desc)
    theta_tensor_desc = torch.FloatTensor(theta_desc)

    ref_x_tensor_desc = torch.FloatTensor(ref_x_desc)
    ref_y_tensor_desc = torch.FloatTensor(ref_y_desc)
    ref_theta_tensor_desc = torch.FloatTensor(ref_theta_desc)

    time_tensor_desc = torch.FloatTensor(time_desc)

    data_desc = torch.FloatTensor([1,1,np.pi/2])
    for i in range(5000):
        df = descripancy(data_desc)
        kx = df[0]
        lx = df[1]
        ky = df[2]
        ly = df[3]
        ktheta = df[4]
        ltheta = df[5]

        beta_x = kx*torch.exp(lx*time_tensor_desc)
        beta_y = ky*torch.exp(ly*time_tensor_desc)
        beta_theta = ktheta * np.pi/2 * torch.exp(ltheta*time_tensor_desc)

        error_x = torch.relu(torch.abs(x_tensor_desc - ref_x_tensor_desc)-beta_x).mean()
        error_y = torch.relu(torch.abs(y_tensor_desc - ref_y_tensor_desc)-beta_y).mean()
        error_theta = torch.relu(torch.abs(theta_tensor_desc - ref_theta_tensor_desc)-beta_theta).mean()

        error_lx = torch.relu(lx)
        error_ly = torch.relu(ly)
        error_ltheta = torch.relu(ltheta)

        loss_constraint = error_x+error_y+error_theta*100+error_lx+error_ly+error_ltheta
        # loss_minimize = (beta_x*time_tensor_desc).mean()+(beta_y*time_tensor_desc).mean()+(beta_theta*time_tensor_desc).mean()
        loss_minimize = beta_x.mean()+beta_y.mean()+beta_theta.mean()
        loss = loss_constraint*10000+loss_minimize
        if i%10 == 0:
            print(i,loss.item())
        optimizer_desc.zero_grad()
        loss.backward()
        optimizer_desc.step()
        scheduler_desc.step()

    df = descripancy(data_desc)
    kx = df[0].item()
    lx = df[1].item()
    ky = df[2].item()
    ly = df[3].item()
    ktheta = df[4].item()
    ltheta = df[5].item()

    tmp_x_upper = []
    tmp_x_lower = []
    tmp_y_upper = []
    tmp_y_lower = []
    tmp_theta_upper = []
    tmp_theta_lower = []
    tmp_time = []

    plt.figure(1)
    plt.figure(2)
    plt.figure(3)
    plt.figure(4)

    for i in range(len(ref)-1):
        t = i*0.01
        tmp_time.append(t)
        dx = kx*np.exp(lx*t)
        dy = ky*np.exp(ly*t)
        dtheta = ktheta*np.exp(ltheta*t)
        tmp_x_upper.append(ref[i][0]+dx)
        tmp_x_lower.append(ref[i][0]-dx)
        tmp_y_upper.append(ref[i][1]+dy)
        tmp_y_lower.append(ref[i][1]-dy)
        tmp_theta_upper.append(ref[i][2]+dtheta)
        tmp_theta_lower.append(ref[i][2]-dtheta)

    x_counter = []
    y_counter = []
    theta_counter = []
    ref_x_counter = []
    ref_y_counter = []
    ref_theta_counter = []
    error_x_counter = []
    error_y_counter = []
    error_theta_counter = []
    error_cos_counter = []
    error_sin_counter = []
    symmetry_y_counter = []


    error_x_counter_tensor = torch.zeros((0,1),device = 'cuda')
    error_y_counter_tensor = torch.zeros((0,1),device = 'cuda')
    error_cos_counter_tensor = torch.zeros((0,1),device = 'cuda')
    error_sin_counter_tensor = torch.zeros((0,1),device = 'cuda')

    for l in range(50):
        x_init = np.random.uniform(-16,-14)
        y_init = np.random.uniform(-1,1)
        theta_init = np.random.uniform(-np.pi/2,np.pi/2)
        print(j,x_init,y_init,theta_init*180/np.pi)

        r = ode(func1)
        r.set_initial_value([x_init,y_init,theta_init])

        tmp_x = []
        tmp_y = []
        tmp_theta = []
        tmp_traj = [[0,x_init,y_init,theta_init]]
        tmp_ref_x = []
        tmp_ref_y = []
        tmp_ref_theta = []

        error_x_counter_tmp = []
        error_y_counter_tmp = []
        error_theta_counter_tmp = []
        error_cos_counter_tmp = []
        error_sin_counter_tmp = []

        symmetry_y_counter_tmp = []

        counter_flag = False
        k = 0
        for i in range(len(ref)-1):
            t = i*0.01
            tmp_x.append(tmp_traj[i][1])
            tmp_y.append(tmp_traj[i][2])
            tmp_theta.append(tmp_traj[i][3])

            dx = kx*np.exp(lx*t)
            dy = ky*np.exp(ly*t)
            dtheta = ktheta*np.exp(ltheta*t)
            
            symmetry_y = tmp_traj[i][2]-ref[i+1][1]

            error_x = ref[i+1][0]-tmp_traj[i][1]
            error_y = np.abs(ref[i+1][1]-tmp_traj[i][2])
            error_theta = (ref[i+1][2]-tmp_traj[i][3])%(np.pi*2)
            error_theta_cos = np.cos(error_theta)
            error_theta_sin = np.sign(symmetry_y)*np.sin(error_theta)
                        
            error_x_counter_tmp.append([error_x])
            error_y_counter_tmp.append([error_y])
            error_theta_counter_tmp.append([error_theta])
            error_cos_counter_tmp.append([error_theta_cos])
            error_sin_counter_tmp.append([error_theta_sin])
            symmetry_y_counter_tmp.append(symmetry_y)

            data_tmp = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
            u = model(data_tmp)
            vr = u[0].item()
            delta = u[1].item()
            delta = delta * np.sign(symmetry_y)

            val = [0,0,0]
            if vr > 30:
                vr = 30
            elif vr < -0:
                vr = -0

            if delta > np.pi/4: 
                delta = np.pi/4
            elif delta < -np.pi/4:
                delta = -np.pi/4

            beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta)/np.cos(delta))
            val[0] = tmp_traj[i][1]+0.01*vr*np.cos(tmp_traj[i][3]+beta)
            val[1] = tmp_traj[i][2]+0.01*vr*np.sin(tmp_traj[i][3]+beta)
            val[2] = tmp_traj[i][3]+0.01*vr/Lr * np.sin(beta)      

            tmp_traj.append([t+0.01,val[0],val[1],val[2]])
            tmp_ref_x.append(ref[i][0])
            tmp_ref_y.append(ref[i][1])
            tmp_ref_theta.append(ref[i][2])

            if np.abs(val[0] - ref[i+1][0]) > dx or np.abs(val[1] - ref[i+1][1]) > dy or np.abs(val[2] - ref[i+1][2]) > dtheta:
                counter_flag = True

        
        if counter_flag:
            k+=1
            x_counter = x_counter+tmp_x
            y_counter = y_counter+tmp_y
            theta_counter = theta_counter+tmp_theta

            symmetry_y_counter = symmetry_y_counter+symmetry_y_counter_tmp

            ref_x_counter = ref_x_counter + tmp_ref_x
            ref_y_counter = ref_y_counter + tmp_ref_y
            ref_theta_counter = ref_theta_counter + tmp_ref_theta

            error_x_counter = error_x_counter + error_x_counter_tmp
            error_y_counter = error_y_counter + error_y_counter_tmp
            error_theta_counter = error_theta_counter + error_theta_counter_tmp
            error_cos_counter = error_cos_counter + error_cos_counter_tmp
            error_sin_counter = error_sin_counter + error_sin_counter_tmp

            plt.figure(1)
            plt.plot(tmp_x,tmp_y,'g')

            plt.figure(2)
            plt.plot(tmp_time,tmp_x,'g')

            plt.figure(3)
            plt.plot(tmp_time,tmp_y,'g')

            plt.figure(4)
            plt.plot(tmp_time,tmp_theta,'g')
            
        else:
            plt.figure(1)
            plt.plot(tmp_x,tmp_y,'b')

            plt.figure(2)
            plt.plot(tmp_time,tmp_x,'b')

            plt.figure(3)
            plt.plot(tmp_time,tmp_y,'b')

            plt.figure(4)
            plt.plot(tmp_time,tmp_theta,'b')

    plt.figure(1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('x vs y')
    fn = './result/fig_xy_'+str(j)+'.png'
    plt.savefig(fn)
    plt.clf()
    
    plt.figure(2)
    plt.plot(tmp_time,tmp_x_upper,'r')
    plt.plot(tmp_time,tmp_x_lower,'r')
    plt.ylim((-20,1))
    plt.xlabel('time')
    plt.ylabel('x')
    plt.title('time vs x')
    fn = './result/fig_tx_'+str(j)+'.png'
    plt.savefig(fn) 
    plt.clf()
    
    plt.figure(3)
    plt.plot(tmp_time,tmp_y_upper,'r')
    plt.plot(tmp_time,tmp_y_lower,'r')
    plt.ylim((-18,18))
    plt.xlabel('time')
    plt.ylabel('y')
    plt.title('time vs y')
    fn = './result/fig_ty_'+str(j)+'.png'
    plt.savefig(fn) 
    plt.clf()
    
    plt.figure(4)
    plt.plot(tmp_time,tmp_theta_upper,'r')
    plt.plot(tmp_time,tmp_theta_lower,'r')
    plt.ylim((-8,8))
    plt.xlabel('time')
    plt.ylabel('theta')
    plt.title('time vs theta')
    fn = './result/fig_ttheta_'+str(j)+'.png'
    plt.savefig(fn) 
    plt.clf()

    plt.figure(5)
    plt.plot(tmp_time,tmp_x_upper,color=cm.cool(j*20))
    plt.plot(tmp_time,tmp_x_lower,color=cm.cool(j*20))
    
    plt.figure(6)
    plt.plot(tmp_time,tmp_y_upper,color=cm.cool(j*20))
    plt.plot(tmp_time,tmp_y_lower,color=cm.cool(j*20))

    plt.figure(7)
    plt.plot(tmp_time,tmp_theta_upper,color=cm.cool(j*20))
    plt.plot(tmp_time,tmp_theta_lower,color=cm.cool(j*20))

    print("feedback num:",k)
    tmp_error_x = torch.FloatTensor(error_x_counter)
    tmp_error_x = tmp_error_x.to('cuda')
    tmp_error_y = torch.FloatTensor(error_y_counter)
    tmp_error_y = tmp_error_y.to('cuda')
    tmp_error_cos = torch.FloatTensor(error_cos_counter)
    tmp_error_cos = tmp_error_cos.to('cuda')
    tmp_error_sin = torch.FloatTensor(error_sin_counter)
    tmp_error_sin = tmp_error_sin.to('cuda')
    
    error_x_counter_tensor = torch.cat((error_x_counter_tensor,tmp_error_x),dim = 0)
    error_y_counter_tensor = torch.cat((error_y_counter_tensor,tmp_error_y),dim = 0)
    error_cos_counter_tensor = torch.cat((error_cos_counter_tensor,tmp_error_cos),dim = 0)
    error_sin_counter_tensor = torch.cat((error_sin_counter_tensor,tmp_error_sin),dim = 0)

    tmp_symmetry_y = torch.FloatTensor(symmetry_y_counter)
    tmp_symmetry_y = tmp_symmetry_y.to('cuda')
    symmetry_y_tensor = torch.cat((symmetry_y_tensor,tmp_symmetry_y),dim = 0)

    tmp_x = torch.FloatTensor(x_counter)
    tmp_x = tmp_x.to('cuda')
    tmp_y = torch.FloatTensor(y_counter)
    tmp_y = tmp_y.to('cuda')
    tmp_theta = torch.FloatTensor(theta_counter)
    tmp_theta = tmp_theta.to('cuda')
    x_tensor = torch.cat((x_tensor,tmp_x),dim = 0)
    y_tensor = torch.cat((y_tensor,tmp_y),dim = 0)
    theta_tensor = torch.cat((theta_tensor,tmp_theta),dim = 0)
            
    tmp_ref_x = torch.FloatTensor(ref_x_counter)
    tmp_ref_y = torch.FloatTensor(ref_y_counter)
    tmp_ref_theta = torch.FloatTensor(ref_theta_counter)
    tmp_ref_x = tmp_ref_x.to('cuda')
    tmp_ref_y = tmp_ref_y.to('cuda')
    tmp_ref_theta = tmp_ref_theta.to('cuda')
    
    ref_x_tensor = torch.cat((ref_x_tensor,tmp_ref_x),dim = 0)
    ref_y_tensor = torch.cat((ref_y_tensor,tmp_ref_y),dim = 0)
    ref_theta_tensor = torch.cat((ref_theta_tensor,tmp_ref_theta),dim = 0)

    data_counter = torch.cat((error_x_counter_tensor,error_y_counter_tensor,error_cos_counter_tensor,error_sin_counter_tensor),dim = 1)
    data = torch.cat((data,data_counter),dim = 0)

    device = torch.device('cuda')
    model = model.to(device)

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

    # symmetry_y_counter = torch.zeros((0,1),device = device)
    
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
    #     error_y_counter_tmp = torch.abs(y_ref_counter_tmp - y_counter_tmp)
    #     error_y_counter = torch.cat((error_y_counter,error_y_counter_tmp),dim = 0)
    #     error_theta_counter_tmp = torch.cos(theta_ref_counter_tmp) - torch.cos(theta_counter_tmp)
    #     error_theta_counter = torch.cat((error_theta_counter,error_theta_counter_tmp),dim = 0)
    #     error_pos_init_tmp = torch.sqrt(error_x_counter_tmp**2+error_y_counter_tmp**2)
    #     error_pos_init = torch.cat((error_pos_init,error_pos_init_tmp),dim = 0)

    #     symmetry_y_counter_tmp = y_counter_tmp - y_ref_counter_tmp
    #     symmetry_y_counter = torch.cat((symmetry_y_counter,symmetry_y_counter_tmp),dim = 0)
    # x_counter = torch.rand(n_counter,1,device = device)*6-3-0.01
    # y_counter = torch.rand(n_counter,1,device = device)*6-3
    # theta_counter = torch.rand(n_counter,1,device = device)*np.pi-np.pi/2

    # error_x_counter = 0-x_counter
    # error_y_counter = 0-y_counter
    # error_theta_counter = (0-theta_counter)%(np.pi*2)
    # error_pos_init = torch.sqrt(error_x_counter**2+error_y_counter**2)


    # data_counter = torch.cat((error_x_counter,error_y_counter,error_theta_counter),dim = 1)
    # control_tensor = model(data_counter)
    # vr = torch.clamp(control_tensor[:,0],-30,30)
    # delta = control_tensor[:,1] * torch.sign(symmetry_y_counter)[:,0]
    # delta = torch.clamp(delta,-np.pi/4,np.pi/4)
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
    #         symmetry_y_tensor = torch.cat((symmetry_y_tensor,symmetry_y_counter[i]),dim = 0)

    # print(j,l)

plt.figure(5)
plt.ylim((-20,1))
plt.xlabel('time')
plt.ylabel('x')
plt.title('desc for x')
plt.savefig('./result/desc_x.png')
plt.clf()

plt.figure(6)
plt.ylim((-18,18))
plt.xlabel('time')
plt.ylabel('y')
plt.title('desc for y')
plt.savefig('./result/desc_y.png')
plt.clf()

plt.figure(7)
plt.ylim((-8,8))
plt.xlabel('time')
plt.ylabel('theta')
plt.title('desc for theta')
plt.savefig('./result/desc_theta.png')
plt.clf()

plt.close('all')

print(data.shape)
device = torch.device('cpu')
model = model.to(device)
for i in range(10):
    x_init = np.random.uniform(-16,-15)
    y_init = np.random.uniform(-1,1)
    # x_init = -15
    # y_init = 0
    theta_init = np.random.uniform(-np.pi/2,np.pi/2)

    trajectory = [[0,x_init,y_init,theta_init]]
    r = ode(func1)
    r.set_initial_value([x_init,y_init,theta_init])

    for i in range(len(ref)-1):
        error_x = ref[i+1][0]-trajectory[i][1]
        error_y = np.abs(ref[i+1][1]-trajectory[i][2])
        symmetry_y = trajectory[i][2]-ref[i+1][1]
        error_theta = ref[i+1][2]-trajectory[i][3]
        error_theta_cos = np.cos(error_theta)
        error_theta_sin = np.sign(symmetry_y)*np.sin(error_theta)

        data = torch.FloatTensor([error_x,error_y,error_theta_cos,error_theta_sin])
        u = model(data)
        vr = u[0].item()
        delta = u[1].item()
        delta = delta * np.sign(symmetry_y)

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

torch.save(model.state_dict(), './model_controller_sym')
torch.save(descripancy.state_dict(),'./desripancy')

# new_x_tensor = new_x_tensor.to(device)
# x_end = new_x_tensor.tolist()
# new_y_tensor = new_y_tensor.to(device)
# y_end = new_y_tensor.tolist()
# new_theta_tensor = new_theta_tensor.to(device)
# theta_end = new_theta_tensor.tolist()

# for i in range(n_sample):
#     plt.plot([sample_x[i],x_end[i]],[sample_y[i],y_end[i]],'b')
#     plt.plot(sample_x[i],sample_y[i],'g.')
#     plt.plot(x_end[i],y_end[i],'r.')

# plt.plot(0,0,'y.')
# plt.show()