import sys
sys.path.append("./utils")
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from torch.autograd import Variable
#from robot import Robot
from logger import Logger
from robot_stone import Robot
import heightmap

import math3d as m3d
import random
from math import *
import pcpt_scoop_res
import scipy
from torch import nn
from PIL import Image


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

htmap_w = 200
htmap_h = 200
workspace_limits_raw = np.array([[-0.01, 0.12], [0.63, 0.76], [0.02, 0.08]])
workspace_limits_list = [np.array([[-0.01, 0.055], [0.63, 0.695], [0.02, 0.08]]), np.array([[0.0225, 0.0875], [0.63, 0.695], [0.02, 0.08]]), np.array([[0.055, 0.12], [0.63, 0.695], [0.02, 0.08]]), np.array([[-0.01, 0.055], [0.6625, 0.7275], [0.02, 0.08]]), np.array([[0.0225, 0.0875], [0.6625, 0.7275], [0.02, 0.08]]), np.array([[0.055, 0.12], [0.6625, 0.7275], [0.02, 0.08]]), np.array([[-0.01, 0.055], [0.695, 0.76], [0.02, 0.08]]), np.array([[0.0225, 0.0875], [0.695, 0.76], [0.02, 0.08]]), np.array([[0.055, 0.12], [0.695, 0.76], [0.02, 0.08]])]
net = pcpt_scoop_res.ResNet(pcpt_block=pcpt_scoop_res.BasicBlock, pcpt_layers=[1,5,1], scoop_block=pcpt_scoop_res.BasicBlock, scoop_layers=[1,5,1], h=htmap_h, w=htmap_w).cuda()     # define the network
net.load_state_dict(torch.load('net_20210503.pkl'))  

def point_position_after_rotation(current_xy, rotation_pole, desired_angle):
    desired_angle_rad = desired_angle*pi/180
    current_displacement = list(np.array(current_xy)-np.array(rotation_pole))
    rotation_matrix = [[cos(desired_angle_rad), -sin(desired_angle_rad)],[sin(desired_angle_rad), cos(desired_angle_rad)]]
    current_displacement = np.expand_dims(current_displacement, axis=1)
    temp = np.dot(rotation_matrix, current_displacement)
    xy_after_rotate = [list(rotation_pole[0]+temp[0])[0], list(rotation_pole[1]+temp[1])[0]]
    return xy_after_rotate 

def from_angle_index_to_gripper_rot_z(angle_index):
    if angle_index>=0 and angle_index<=4:
        rot_z = (-pi/4)*angle_index
    elif angle_index==7:
        rot_z = pi/4
    elif angle_index==6:
        rot_z = pi/2
    elif angle_index==5:
        rot_z = pi*3/4
    return rot_z

def from_pixel_to_world_position(pixel, workspace_limits, heightmap_resolution, depth_heightmap):
    htmap_h = int(round((workspace_limits[1][1]-workspace_limits[1][0])/heightmap_resolution))
    htmap_w = int(round((workspace_limits[0][1]-workspace_limits[0][0])/heightmap_resolution))
    pix_x = pixel[0]
    pix_y = pixel[1]
    pos = [pix_x * heightmap_resolution + workspace_limits[0][0], pix_y * heightmap_resolution + workspace_limits[1][0], depth_heightmap[pix_y][pix_x] + workspace_limits[2][0]]
    return pos

def main():
    robot = Robot("192.168.1.102")
    while True:
        while True:
            whether_continue = input('Whether continue? (y or n) Shuffle!')
            if whether_continue=='y':
                break
        max_score_pixel_list = []
        color_img_array, depth_array =  robot.getCameraData()
        for workspace_limits in workspace_limits_list:
            color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img_array, depth_array, robot.cam_intrinsics, robot.baseTcam, workspace_limits, robot.heightmap_resolution)
            color_heightmap_image = Image.fromarray(color_heightmap)
            depth_heightmap_image = Image.fromarray((depth_heightmap/0.08*255).astype(np.uint8))
            for rotate_index in range(8):
                rotated_color_heightmap = np.array(color_heightmap_image.rotate(angle = rotate_index*45, fillcolor = (0,0,0))).astype(np.uint8) 
                rotated_depth_heightmap = np.array(depth_heightmap_image.rotate(angle = rotate_index*45, fillcolor = 0)).astype(np.uint8)
                color_depth_heightmap = np.concatenate((rotated_color_heightmap[:,:,[2,1,0]],rotated_depth_heightmap[:, :, np.newaxis]), axis = 2)[np.newaxis, :, :, :]
                color_depth_heightmap = torch.from_numpy(color_depth_heightmap/255.0).permute(0,3,1,2).cuda().float()
                with torch.no_grad():
                    net.eval()
                    predicted_score = net(color_depth_heightmap).permute(0,2,3,1)
                    soft_max_function = nn.Softmax(dim=3)
                    predicted_score = soft_max_function(predicted_score)[:,:,:,1]
                    predicted_score = torch.reshape(predicted_score, (predicted_score.shape[1], predicted_score.shape[2])).cpu().numpy()
                torch.cuda.empty_cache()
                max_score_pixel = np.unravel_index(np.argmax(predicted_score), predicted_score.shape)
                max_score_pixel = [max_score_pixel[1], max_score_pixel[0]]
                max_score_pixel = point_position_after_rotation([max_score_pixel[0], htmap_h-max_score_pixel[1]], [htmap_w/2, htmap_h/2], -rotate_index*45)   # plus or minus
                max_score_pixel = [int(max_score_pixel[0]), int(htmap_h-max_score_pixel[1])]
                pos = from_pixel_to_world_position(max_score_pixel, workspace_limits, robot.heightmap_resolution, depth_heightmap)
                yaw = from_angle_index_to_gripper_rot_z(rotate_index)
                max_score_pixel_list.append([pos, yaw, np.max(predicted_score)])

        max_score_pixel_list.sort(key=lambda x: x[2], reverse=True)
        ini_aperture = 0.02
        for max_score_pixel in max_score_pixel_list:
            pos = max_score_pixel[0]
            yaw = max_score_pixel[1]
            print('pos: ', pos, ' yaw: ', yaw, ' aperture: ', ini_aperture)
            color_heightmap_raw, _ = heightmap.get_heightmap(color_img_array, depth_array, robot.cam_intrinsics, robot.baseTcam, workspace_limits_raw, robot.heightmap_resolution)
            copy_color_heightmap = color_heightmap_raw.copy()
            pix_x = int(max(min(round((pos[0] - workspace_limits_raw[0][0])/robot.heightmap_resolution), 400-1), 0))
            pix_y = int(max(min(round((pos[1] - workspace_limits_raw[1][0])/robot.heightmap_resolution), 400-1), 0))
            cv2.circle(copy_color_heightmap, (pix_x,pix_y), 5, (255, 0, 0), 3)
            pix_x_prime = int(max(min(round((pos[0]-0.02*sin(yaw) - workspace_limits_raw[0][0])/robot.heightmap_resolution), 400-1), 0))
            pix_y_prime = int(max(min(round((pos[1]-0.02*cos(yaw) - workspace_limits_raw[1][0])/robot.heightmap_resolution), 400-1), 0))
            cv2.arrowedLine(copy_color_heightmap, (pix_x_prime, pix_y_prime), (pix_x, pix_y), (0, 0, 255), 2)
            test_image = cv2.flip(copy_color_heightmap, 1)
            #cv2.imshow("visualization",test_image*255)
            f = plt.figure(1)
            plt.imshow(test_image[:,:,[2,1,0]])
            plt.show()
            grasp_success = robot.exe_scoop(pos, yaw, ini_aperture)
            if grasp_success==-1:
                continue
            else:
                break

if __name__ == '__main__':
    main()
    '''
    kkk = cv2.imread('/home/terry/catkin_ws/src/dg_learning_real_one_net/picture_20210422/depth_heightmap/8_depth_heightmap.jpg', -1)
    max_score_pixel = np.unravel_index(np.argmax(kkk), kkk.shape)
    max_score_pixel = [max_score_pixel[1], max_score_pixel[0]]
    print('max_score_pixel', max_score_pixel)
    kkk_image = Image.fromarray(kkk)
    rotated_kkk = np.array(kkk_image.rotate(angle = 45, fillcolor = 0)).astype(np.uint8) 
    max_score_pixel = np.unravel_index(np.argmax(rotated_kkk), rotated_kkk.shape)
    max_score_pixel = [max_score_pixel[1], max_score_pixel[0]]
    print('max_score_pixel', max_score_pixel)
    max_score_pixel = point_position_after_rotation([max_score_pixel[0], 400-max_score_pixel[1]], [200, 200], -45)
    max_score_pixel = [max_score_pixel[0], 400-max_score_pixel[1]]
    print('max_score_pixel', max_score_pixel)
    '''
