import torch
import numpy as np
import random
import matplotlib.pyplot as plt

#############################
data_straight = []
input_straight = []
output_straight = []
x_straight = []
y_straight = []

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

for j in range(0,1):
    for k in range(0,360,5):
        data_temp = []
        with open("./data/data_straight"+str(int(k))+"_"+str(int(j))+".dat") as file:
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [float(i) for i in line]
                data_temp.append(line)
                line = file.readline()

            # if k==30:
            #     for i in range(len(data_temp)):
            #         x_straight.append(data_temp[i][1])
            #         y_straight.append(data_temp[i][2])

            #     plt.plot(x_straight,y_straight)
            #     plt.title("Turning Angle delta_f=0")
            #     plt.xlabel("Position x")
            #     plt.ylabel("Position y")
            #     plt.show()

            input_temp = []
            for i in range(0,len(data_temp)-1):
                input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5],data_temp[i][6]])

            output_temp = []
            for i in range(1,len(data_temp)):
                temp = []
                # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
                temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
                # temp.append(((data_temp[i][3]-data_temp[i-1][3])/delta_t)%int(360))
                output_temp.append(temp)

            data_straight = data_straight + data_temp
            input_straight = input_straight + input_temp
            output_straight = output_straight + output_temp


#############################
data_pos = []
input_pos = []
output_pos = []
for k in range(0,1):
    for j in range(10,31,10):
        x_curve = []
        y_curve = []
        data_temp = []
        with open("./data/data_pos"+str(int(j))+"_"+str(int(k))+".dat") as file:
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [float(i) for i in line]
                data_temp.append(line)
                line = file.readline()

        for i in range(len(data_temp)):
            x_curve.append(data_temp[i][1])
            y_curve.append(data_temp[i][2])

        input_temp = []
        for i in range(0,len(data_temp)-1):
            input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5],data_temp[i][6]])

        output_temp = []
        for i in range(1,len(data_temp)):
            temp = []
            # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
            temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
            # temp.append(((data_temp[i][3]-data_temp[i-1][3])/delta_t)%int(360))
            output_temp.append(temp)

        data_pos = data_pos + data_temp
        input_pos = input_pos + input_temp
        output_pos = output_pos + output_temp

        # plt.plot(x_curve,y_curve)
        # plt.title("Turning Angle delta_f="+str(int(j)))
        # plt.xlabel("Position x")
        # plt.ylabel("Position y")
        # plt.show()

#############################
data_neg = []
input_neg = []
output_neg = []
for k in range(0,1):
    for j in range(10,31,10):
        x_curve = []
        y_curve = []
        data_temp = []
        with open("./data/data_neg"+str(int(j))+"_"+str(int(k))+".dat") as file:
            line = file.readline()
            while line:
                line = line.split(' ')
                line = [float(i) for i in line]
                data_temp.append(line)
                line = file.readline()

        for i in range(len(data_temp)):
            x_curve.append(data_temp[i][1])
            y_curve.append(data_temp[i][2])

        input_temp = []
        for i in range(0,len(data_temp)-1):
            input_temp.append([(data_temp[i][3]*180/np.pi) % int(360),data_temp[i][4],data_temp[i][5],data_temp[i][6]])

        output_temp = []
        for i in range(1,len(data_temp)):
            temp = []
            # temp.append((data_temp[i][1]-data_temp[i-1][1])/delta_t)
            temp.append((data_temp[i][2]-data_temp[i-1][2])/delta_t)
            # temp.append(((data_temp[i][3]-data_temp[i-1][3])/delta_t)%int(360))
            output_temp.append(temp)

        plt.plot(x_curve,y_curve)
        plt.title("Turning Angle delta_f=-"+str(int(j)))
        plt.xlabel("Position x")
        plt.ylabel("Position y")
        plt.show()

        data_neg = data_neg + data_temp
        input_neg = input_neg + input_temp
        output_neg = output_neg + output_temp
#############################
