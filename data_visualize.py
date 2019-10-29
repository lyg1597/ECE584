import matplotlib.pyplot as plt
import numpy as np

filename = "data_pos30_0.dat"
delta_t = 0.01

data = []
input_data = []
output_data = []
for i in range(4):
    data_temp = []
    with open("data_pos30_"+str(int(i))+".dat") as file:
        line = file.readline()
        while line:
            line = line.split(' ')
            line = [float(i) for i in line]
            data_temp.append(line)
            line = file.readline()

    input_temp = []
    for i in range(0,len(data_temp)-1):
        input_temp.append([data_temp[i][1],data_temp[i][2],(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5]])

    output_temp = []
    for i in range(1,len(data_temp)):
        temp = []
        temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
        temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
        temp.append(((data_temp[i][3]-data_temp[i-1][3])/delta_t)%int(360))
        output_temp.append(temp)
    
    data = data + data_temp
    input_data = input_data + input_temp
    output_data = output_data + output_temp

x = []
y = []
theta = []
v = []
delta = []
for i in range(len(input_data)):
    x.append(input_data[i][0])
    y.append(input_data[i][1])
    theta.append(input_data[i][2])
    v.append(input_data[i][3])
    v.append(input_data[i][4])

dx = []
dy = []
dtheta = []
for i in range(len(output_data)):
    dx.append(output_data[i][0])
    dy.append(output_data[i][1])
    dtheta.append(output_data[i][2])

plt.plot(theta,dy,'bo')
plt.show()

plt.plot(theta,dx,'bo')
plt.show()

plt.plot(dtheta,'bo')
plt.show()
