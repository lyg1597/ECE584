import matplotlib.pyplot as plt
import numpy as np

filename = "data_pos30_0.dat"
delta_t = 0.01

data = []
with open(filename) as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data.append(line)
        line = file.readline()

input_data = []
for i in range(0,len(data)-1):
    input_data.append([(data[i][3]*180/np.pi) % int(360),data[i][4],data[i][5]])

output_data = []
for i in range(1,len(data)):
    temp = []
    temp.append((data[i][1]-data[i-1][1])/delta_t)
    temp.append((data[i][2]-data[i-1][2])/delta_t)
    temp.append(((data[i][3]-data[i-1][3])/delta_t)%int(360))
    output_data.append(temp)

x = []
y = []
theta = []
for i in range(len(data)):
    x.append(data[i][1])
    y.append(data[i][2])
    theta.append((data[i][3]*180/np.pi)%int(360))

plt.plot(x,y)
plt.show()

