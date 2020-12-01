import torch
from scipy.integrate import ode
import numpy as np
import polytope as pc
from typing import Optional, List, Tuple
import math
import matplotlib.pyplot as plt

class Waypoint:
    def __init__(self, mode: str, mode_parameters: List[float], time_bound: float, id: int,
                 unsafeset_list = None):
        self.mode: str = mode
        self.mode_parameters: List[float] = mode_parameters
        self.time_bound: float = time_bound
        self.id = id
        self.unsafeset_list = unsafeset_list

    def is_equal(self, other_waypoint: List[float]):
        return tuple(self.mode_parameters) == tuple(other_waypoint.mode_parameters)
        # self.delta: np.array = (self.original_guard[1, :] - self.original_guard[0, :]) / 2
    # TODO add helper function to check if point is inside guard

def func1(t, vars, u):
    curr_x = vars[0]
    curr_y = vars[1]
    curr_theta = vars[2]
    vr = u[0]
    delta = u[1]

    Lr = 2
    Lf = 2
    beta = np.arctan(Lr/(Lr+Lf)*np.sin(delta)/np.cos(delta))
    dx = vr*np.cos(curr_theta+beta)
    dy = vr*np.sin(curr_theta+beta)
    dtheta = vr/Lr * np.sin(beta)
    return [dx, dy, dtheta]

def lidarSimulator(state, obstacles, range = 50, resolution = 0.05, scan_number = 360):
    x = state[0]
    y = state[1]
    theta = state[2]
    
    scan_angle_list = np.linspace(0, np.pi*2, scan_number, endpoint = False)
    base_scan_vector_x = np.expand_dims(np.arange(0,range,resolution), axis = 1)
    base_scan_vector_y = np.zeros(base_scan_vector_x.shape)
    # base_scan_vector = np.concatenate((base_scan_vector_x, base_scan_vector_y), axis = 1)
    point_cloud = []
    for scan_angle in scan_angle_list:
        # pass
        scan_vector_x = (np.cos(scan_angle + theta) * base_scan_vector_x - np.sin(scan_angle + theta) * base_scan_vector_y) + x
        scan_vector_y = (np.sin(scan_angle + theta) * base_scan_vector_x + np.cos(scan_angle + theta) * base_scan_vector_y) + y
        scan_vector = np.concatenate((scan_vector_x, scan_vector_y), axis = 1)
        idx = scan_vector.shape[0]-1
        for obstacle in obstacles:
            res = obstacle.contains(scan_vector.T)
            if np.any(res):
                val = np.argwhere(res==True)[0][0]
                if val < idx:
                    idx = val
        pt_x = (np.cos(-theta)*(scan_vector_x[idx-1]-x) - np.sin(-theta)*(scan_vector_y[idx-1]-y))
        pt_y = (np.sin(-theta)*(scan_vector_x[idx-1]-x) + np.cos(-theta)*(scan_vector_y[idx-1]-y))
        point_cloud.append([pt_x, pt_y])
    return np.array(point_cloud)

def convertToWorld(state, point_cloud):
    x = state[0]
    y = state[1]
    theta = state[2]

    point_x = point_cloud[:,0]
    point_y = point_cloud[:,1]

    world_x = (np.cos(theta)*point_x - np.sin(theta)*point_y) + x
    world_y = (np.sin(theta)*point_x + np.cos(theta)*point_y) + y 
    world_point_cloud = np.concatenate((world_x, world_y), axis = 1)
    return world_point_cloud

def checkObstacleFront(data_points, curr_state, waypoint):
    curr_x = curr_state[0]
    curr_y = curr_state[1]

    # First box
    [x2, y2] = waypoint.mode_parameters[0:2]
    theta = np.arctan2(y2-curr_y, x2-curr_x)
    x_tmp = 0.5
    y_tmp = 0
    dx = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    transform_vector = np.array([[dx,dy],[dx,dy],[-dx,-dy],[-dx,-dy]])
    center_vector = np.array([[x2,y2],[curr_x,curr_y],[curr_x,curr_y],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res1 = poly.contains(data_points[:,0:2].T)
    res1 = np.any(res1)

    # Second box
    [x1, y1] = waypoint.mode_parameters[0:2]
    [x2, y2] = waypoint.mode_parameters[2:4]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 1
    y_tmp = 0
    dx = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    transform_vector = np.array([[dx,dy],[dx,dy],[-dx,-dy],[-dx,-dy]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res2 = poly.contains(data_points[:,0:2].T)
    res2 = np.any(res2)

    # Third box
    [x1, y1] = waypoint.mode_parameters[2:4]
    [x2, y2] = waypoint.mode_parameters[4:6]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 1
    y_tmp = 0
    dx = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    transform_vector = np.array([[dx,dy],[dx,dy],[-dx,-dy],[-dx,-dy]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res3 = poly.contains(data_points[:,0:2].T)
    res3 = np.any(res3)

    return int(res1 or res2 or res3)

def checkObstacleFrontLeft(data_points, curr_state, waypoint):
    # Second box
    [x1, y1] = waypoint.mode_parameters[0:2]
    [x2, y2] = waypoint.mode_parameters[2:4]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 4
    y_tmp = 0
    dx1 = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy1 = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    x_tmp = 2
    y_tmp = 0
    dx2 = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy2 = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    transform_vector = np.array([[dx1,dy1],[dx1,dy1],[dx2,dy2],[dx2,dy2]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res2_left = poly.contains(data_points[:,0:2].T)
    res2_left = np.any(res2_left)

    # Third box 
    [x1, y1] = waypoint.mode_parameters[2:4]
    [x2, y2] = waypoint.mode_parameters[4:6]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 4
    y_tmp = 0
    dx1 = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy1 = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    x_tmp = 2
    y_tmp = 0
    dx2 = x_tmp*np.cos(theta+np.pi/2) - y_tmp*np.sin(theta+np.pi/2)
    dy2 = x_tmp*np.sin(theta+np.pi/2) + y_tmp*np.cos(theta+np.pi/2)
    transform_vector = np.array([[dx1,dy1],[dx1,dy1],[dx2,dy2],[dx2,dy2]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res3_left = poly.contains(data_points[:,0:2].T)
    res3_left = np.any(res3_left)

    return int(res2_left or res3_left)

def checkObstacleFrontRight(data_points, curr_state, waypoint):
    [x1, y1] = waypoint.mode_parameters[0:2]
    [x2, y2] = waypoint.mode_parameters[2:4]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 4
    y_tmp = 0
    dx1 = x_tmp*np.cos(theta-np.pi/2) - y_tmp*np.sin(theta-np.pi/2)
    dy1 = x_tmp*np.sin(theta-np.pi/2) + y_tmp*np.cos(theta-np.pi/2)
    x_tmp = 2
    y_tmp = 0
    dx2 = x_tmp*np.cos(theta-np.pi/2) - y_tmp*np.sin(theta-np.pi/2)
    dy2 = x_tmp*np.sin(theta-np.pi/2) + y_tmp*np.cos(theta-np.pi/2)
    transform_vector = np.array([[dx1,dy1],[dx1,dy1],[dx2,dy2],[dx2,dy2]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res2_right = poly.contains(data_points[:,0:2].T)
    res2_right = np.any(res2_right)

    [x1, y1] = waypoint.mode_parameters[2:4]
    [x2, y2] = waypoint.mode_parameters[4:6]
    theta = np.arctan2(y2-y1, x2-x1)
    x_tmp = 4
    y_tmp = 0
    dx1 = x_tmp*np.cos(theta-np.pi/2) - y_tmp*np.sin(theta-np.pi/2)
    dy1 = x_tmp*np.sin(theta-np.pi/2) + y_tmp*np.cos(theta-np.pi/2)
    x_tmp = 2
    y_tmp = 0
    dx2 = x_tmp*np.cos(theta-np.pi/2) - y_tmp*np.sin(theta-np.pi/2)
    dy2 = x_tmp*np.sin(theta-np.pi/2) + y_tmp*np.cos(theta-np.pi/2)
    transform_vector = np.array([[dx1,dy1],[dx1,dy1],[dx2,dy2],[dx2,dy2]])
    center_vector = np.array([[x2,y2],[x1,y1],[x1,y1],[x2,y2]])
    vertices = center_vector + transform_vector
    poly = pc.qhull(vertices)
    res3_right = poly.contains(data_points[:,0:2].T)
    res3_right = np.any(res3_right)

    return int(res2_right or res3_right)

def runModel(waypoint, time_step, initial_point, time_bound):
    init = initial_point
    trajectory = [init]
    r = ode(func1)
    r.set_initial_value(init)
    t = 0
    target_x = waypoint.mode_parameters[0]
    target_y = waypoint.mode_parameters[1]
    i = 0

    # Get lidar reading
    point_cloud = lidarSimulator(trajectory[i], waypoint.unsafeset_list)
    point_cloud = convertToWorld(trajectory[i], point_cloud)
    res_front = checkObstacleFront(point_cloud, trajectory[i], waypoint)
    res_front_left = checkObstacleFrontLeft(point_cloud, trajectory[i], waypoint)
    res_front_right = checkObstacleFrontRight(point_cloud, trajectory[i], waypoint)
    trace = [[t]]
    trace[i].extend(trajectory[i])
    trace[i].extend([res_front, res_front_left, res_front_right])

    while t <= time_bound:
        ex = (target_x - trajectory[i][0])*np.cos(trajectory[i][2]) + (target_y - trajectory[i][1])*np.sin(trajectory[i][2])
        ey = -(target_x - trajectory[i][0])*np.sin(trajectory[i][2]) + (target_y - trajectory[i][1])*np.cos(trajectory[i][2])
        
        k_s = 0.1
        k_n = 0.1
        v = ex * 3
        delta = ey * 1
        u = [v,delta]
        r.set_f_params(u)
        val = r.integrate(r.t + time_step)

        trajectory.append(val.tolist())

        t += time_step
        i += 1

        # Get lidar reading
        point_cloud = lidarSimulator(trajectory[i], waypoint.unsafeset_list)
        point_cloud = convertToWorld(trajectory[i], point_cloud)
        res_front = checkObstacleFront(point_cloud, trajectory[i], waypoint)
        res_front_left = checkObstacleFrontLeft(point_cloud, trajectory[i], waypoint)
        res_front_right = checkObstacleFrontRight(point_cloud, trajectory[i], waypoint)
        trace.append([t])
        trace[i].extend(trajectory[i])
        trace[i].extend([res_front, res_front_left, res_front_right])
        print([res_front, res_front_left, res_front_right])

    return trace

def TC_Simulate(waypoint, time_step, initial_point):
    res = runModel(waypoint, time_step, initial_point, waypoint.time_bound)
    return res

if __name__ == "__main__":
    init_x = 0
    init_y = 0
    init_theta = 0
    vertices = np.array([[10,3],[15,3],[15,4],[10,4]])
    poly1 = pc.qhull(vertices)
    vertices = np.array([[10,-3],[15,-3],[15,-4],[10,-4]])
    poly2 = pc.qhull(vertices)
    waypoint = Waypoint("follow_waypoint", [5,0,10,0,15,0], 10, 0, [poly1, poly2])
    init_point = [init_x, init_y, init_theta]
    res = TC_Simulate(waypoint, 0.01, init_point)
    print(res)
    
    # state = [0,1,np.pi/2]
    # vertices = np.array([[10,-5],[20,-5],[20,5],[10,5]])
    # poly1 = pc.qhull(vertices)
    # vertices = np.array([[0,10],[5,15],[0,20],[-5,15]])
    # poly2 = pc.qhull(vertices)

    # point_cloud = lidarSimulator(state,[poly1, poly2])
    # plt.plot(point_cloud[:,0],point_cloud[:,1],'.')
    # plt.show()
    # point_cloud = convertToWorld(state, point_cloud)
    # plt.plot(point_cloud[:,0],point_cloud[:,1],'.')
    # plt.show()
    
