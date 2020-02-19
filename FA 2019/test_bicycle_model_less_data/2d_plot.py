import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm

# [pos_x,pos_y,orientation,forward_speed,input_acceleration,input_turning]

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H1,D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        # self.linear2 = torch.nn.Linear(H1, H2)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self,x):
        h1 = torch.nn.functional.relu(self.linear1(x))
        # h2 = torch.nn.functional.relu(self.linear2(h1))
        y = self.linear2(h1)
        return y

delta_t = 0.01
delta_const = 30
init_v = 3
a_const = 0
time_horizon = 30
init_x = 0
init_y = 0
init_theta = 0
Lr = 5
Lf = 3

def getIp(time):
    # if(time<5):
    #     delta = 28
    # elif time>=5 and time<10:
    #     delta = -15
    # elif time>=10 and time<15:
    #     delta = 16
    # elif time>=15 and time<20:
    #     delta = -27
    # else:
    #     delta = 16

    # delta = 0

    a = a_const
    delta_temp = delta_const*np.pi/180
    beta = np.arctan(Lr/(Lr+Lf) * np.sin(delta_temp)/np.cos(delta_temp))
    x = [a,delta,beta]
    return x

def func1(vars,time):
    v = vars[3]
    theta = vars[2]
    ip = getIp(time)
    a=ip[0]
    beta = ip[2]

    dx = v*np.cos(theta+beta)
    dy = v*np.sin(theta+beta)
    dtheta = v/Lr * np.sin(beta)
    dv = a 
    return [dx,dy,dtheta,dv]

theta = np.arange(0,360,0.1)
delta = np.arange(-30,30,0.01)

# theta_grid,delta_grid = np.meshgrid(theta,delta,sparse=True)

X=theta
Y=delta
Z=np.zeros((len(theta),len(delta)))
Zx=np.zeros((len(theta),len(delta)))
Zy=np.zeros((len(theta),len(delta)))
Ztheta=np.zeros((len(theta),len(delta)))
Zpos=np.zeros((len(theta),len(delta)))

model_x = TwoLayerNet(4,20,1)
model_y = TwoLayerNet(4,20,1)
model_theta = TwoLayerNet(4,20,1)
model_v = TwoLayerNet(4,20,1)

model_x.load_state_dict(torch.load('./model_x_more_full_state'))
model_y.load_state_dict(torch.load('./model_y_more_full_state'))
model_theta.load_state_dict(torch.load('./model_theta_more_full_state'))
model_v.load_state_dict(torch.load('./model_v_more_full_state'))

for i in range(len(theta)):
    print(i)
    for j in range(len(delta)):
        theta_val = theta[i]
        delta_val = delta[j]
        state = [0,0,theta_val, 3,0,delta_val]
        data = [state[2:6]]

        x_tensor = torch.FloatTensor(data)
        dx_tensor = model_x(x_tensor)
        dx = dx_tensor.data.tolist()[0][0]

        y_tensor = torch.FloatTensor(data)
        dy_tensor = model_y(y_tensor)
        dy = dy_tensor.data.tolist()[0][0]

        theta_tensor = torch.FloatTensor(data)
        dtheta_tensor = model_theta(theta_tensor)
        dtheta = dtheta_tensor.data.tolist()[0][0]

        v_tensor = torch.FloatTensor(data)
        dv_tensor = model_v(v_tensor)
        dv = dv_tensor.data.tolist()[0][0]

        delta_const = delta_val
        a_const = 0
        timeGrid = np.arange(0,0.015,0.01)
        initR = [0,0,theta_val*np.pi/180,3]
        fR = odeint(func1,initR,timeGrid)
        
        dx_tag = (fR[1,0]-fR[0,0])/0.01
        dy_tag = (fR[1,1]-fR[0,1])/0.01
        dtheta_tag = (fR[1,2]*180/np.pi-fR[0,2]*180/np.pi)/0.01
        dv_tag = (fR[1,3]-fR[0,3])/0.01

        Z[i,j] = np.sqrt((dx-dx_tag)**2+(dy-dy_tag)**2+(dtheta-dtheta_tag)**2+(dv-dv_tag)**2)
        Zx[i,j] = abs(dx_tag - dx)
        Zy[i,j] = abs(dy_tag - dy)
        Ztheta[i,j] = abs(dtheta_tag - dtheta)
        Zpos[i,j] = np.sqrt((dx-dx_tag)**2+(dy-dy_tag)**2)

X, Y = np.meshgrid(Y, X)
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_ylabel('theta/degree')
ax.set_xlabel('delta/degree')
ax.set_zlabel('error')
ax.set_title("Error for all state variable estimation")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Zx, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_ylabel('theta/degree')
ax.set_xlabel('delta/degree')
ax.set_zlabel('error')
ax.set_title("Error for x estimation")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Zy, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_ylabel('theta/degree')
ax.set_xlabel('delta/degree')
ax.set_zlabel('error')
ax.set_title("Error for y estimation")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Ztheta, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_ylabel('theta/degree')
ax.set_xlabel('delta/degree')
ax.set_zlabel('error')
ax.set_title("Error for theta estimation")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Zpos, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax.set_ylabel('theta/degree')
ax.set_xlabel('delta/degree')
ax.set_zlabel('error')
ax.set_title("Error for position estimation")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
