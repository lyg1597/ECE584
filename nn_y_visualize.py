import matplotlib.pyplot as plt
import numpy as np
delta_t = 0.01

#############################
data_straight = []
with open("data_straight.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_straight.append(line)
        line = file.readline()

input_straight = []
for i in range(0,len(data_straight)-1):
    input_straight.append([(data_straight[i][3]*180/np.pi) % int(360),data_straight[i][4],data_straight[i][5]])

output_straight = []
for i in range(1,len(data_straight)):
    temp = []
    # temp.append((data_straight[i][1]-data_straight[i-1][1])/delta_t)
    # temp.append((data_straight[i][2]-data_straight[i-1][2])/delta_t)
    temp.append(((data_straight[i][3]-data_straight[i-1][3])/delta_t)%int(360))
    output_straight.append(temp)

#############################
data_pos30 = []
with open("data_pos30.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_pos30.append(line)
        line = file.readline()

input_pos30 = []
for i in range(0,len(data_pos30)-1):
    input_pos30.append([(data_pos30[i][3]*180/np.pi) % int(360),data_pos30[i][4],data_pos30[i][5]])

output_pos30 = []
for i in range(1,len(data_pos30)):
    temp = []
    # temp.append((data_pos30[i][1]-data_pos30[i-1][1])/delta_t)
    # temp.append((data_pos30[i][2]-data_pos30[i-1][2])/delta_t)
    temp.append(((data_pos30[i][3]-data_pos30[i-1][3])/delta_t)%int(360))
    output_pos30.append(temp)

#############################
data_neg30 = []
with open("data_neg30.dat") as file:
    line = file.readline()
    while line:
        line = line.split(' ')
        line = [float(i) for i in line]
        data_neg30.append(line)
        line = file.readline()

input_neg30 = []
for i in range(0,len(data_neg30)-1):
    input_neg30.append([(data_neg30[i][3]*180/np.pi) % int(360),data_neg30[i][4],data_neg30[i][5]])

output_neg30 = []
for i in range(1,len(data_neg30)):
    temp = []
    # temp.append((data_neg30[i][1]-data_neg30[i-1][1])/delta_t)
    # temp.append((data_neg30[i][2]-data_neg30[i-1][2])/delta_t)
    temp.append(((data_neg30[i][3]-data_neg30[i-1][3])/delta_t)%int(360))
    output_neg30.append(temp)

#############################

x = []
y = []
for i in range(0,len(output_pos30)):
    y.append(output_pos30[i][0])
    x.append(input_pos30[i][2])

plt.plot(y,'o')
plt.show()