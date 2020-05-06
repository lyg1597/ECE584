import numpy as np

delta_t = 0.01
data = []
input_data = []
output_data = []
for j in range(0,8):
    with open("./data_vehicle/data"+str(int(j))+".dat") as file:
        input_temp = []
        output_temp = []
        data_temp = []
        line = file.readline()
        while line:
            line = line.split(' ')
            line = [float(i) for i in line]
            data_temp.append(line)
            line = file.readline()

        for i in range(len(data_temp)-1):
            line = data_temp[i]
            curr_time = line[0]
            curr_x = line[1]
            curr_y = line[2]
            target_x = line[3]
            target_y = line[4]
            curr_theta = line[5]
            curr_delta = line[6]
            curr_v = line[7]
            input_temp.append([curr_theta,curr_delta,curr_v])

            line = data_temp[i]
            next_time = line[0]
            next_x = line[1]
            next_y = line[2]
            target_x = line[3]
            target_y = line[4]
            next_theta = line[5]
            next_delta = line[6]
            next_v = line[7]

            x_diff = (next_x-curr_x)/delta_t
            y_diff = (next_y-curr_y)/delta_t
            theta_diff = ((next_theta-curr_theta)/delta_t)%(2*np.pi)
            if theta_diff > np.pi:
                theta_diff = theta_diff - 2*np.pi
            output_temp.append([x_diff])

        data = data + data_temp
        input_data = input_data + input_temp
        output_data = output_data + output_temp

for i in range(len(input_data)):
    tmp = input_data[i]
    if tmp[1] == np.pi/6 and tmp[0] < 0.05:
        print(i,":",tmp)