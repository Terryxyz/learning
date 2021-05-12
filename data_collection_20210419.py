import sys
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
from Mask_R_CNN_stone import Mask_R_CNN_stone
import random
from math import *




def main():
    robot = Robot("192.168.1.102")
    workspace_limits = robot.workspace_limits
    heightmap_resolution = robot.heightmap_resolution
    htmap_h = int(round((workspace_limits[1][1]-workspace_limits[1][0])/heightmap_resolution))
    htmap_w = int(round((workspace_limits[0][1]-workspace_limits[0][0])/heightmap_resolution))
    mask_r_cnn_stone = Mask_R_CNN_stone()
    data_order = 0

    while(True):
        if not os.path.exists('./data_collection/my_data.txt'):
            data_order += 1
        else:
            if data_order != 0:
                data_order += 1
            else:
                data_order_history = np.loadtxt('./data_collection/data_order.txt')
                if data_order_history.shape == ():
                    data_order = int(data_order_history) + 1
                else:
                    data_order = int(data_order_history[-1]) + 1
        print('data order', data_order)
        while True:
            print('Stop shuffling!')
            color_img, depth_img, color_img_path, depth_img_path = robot.getCameraData(data_order)
            color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.baseTcam, workspace_limits, heightmap_resolution)
            whether_use_MaskRCNN = input("Using Mask-R-CNN? (y or n)")
            if whether_use_MaskRCNN == 'n':
                while True:
                    pix_x = int(max(min(round(np.random.normal(htmap_w/2, htmap_w/4)), htmap_w-1), 0))
                    pix_y = int(max(min(round(np.random.normal(htmap_h/2, htmap_h/4)), htmap_h-1), 0))
                    pos = [pix_x * heightmap_resolution + workspace_limits[0][0], pix_y * heightmap_resolution + workspace_limits[1][0], depth_heightmap[pix_y][pix_x] + workspace_limits[2][0]]
                    if pos[2]<=workspace_limits[2][1]:
                        #print('Noise in heightmap')
                        break
                input_img_rot_index = random.randint(-3, 4)
                yaw = input_img_rot_index * (pi/4)

                ini_aperture_index = random.randint(0,3)
                if ini_aperture_index ==0:
                    ini_aperture = 0.015
                elif ini_aperture_index ==1:
                    ini_aperture = 0.02
                elif ini_aperture_index ==2:
                    ini_aperture = 0.025
                elif ini_aperture_index ==3:
                    ini_aperture = 0.03
                else:
                    raise NameError("Error ini aperture index")
                break
            elif whether_use_MaskRCNN == 'y':
                MaskRCNNResult = mask_r_cnn_stone.Mask_R_CNN_result(color_img_path, depth_img_path, depth_img*1000)
                pos, yaw = robot.fromResultToRobotParameter(MaskRCNNResult)
                if pos==None and yaw == None:
                    print("All the recognized objects are not satisfied! Please shuffle!")
                    while True:
                        whether_finish_shuffle = input("Finish shuffle? (y or n)")
                        if whether_finish_shuffle=='y':
                            break
                    continue
                pix_x = int(max(min(round((pos[0] - workspace_limits[0][0])/heightmap_resolution), htmap_w-1), 0))
                pix_y = int(max(min(round((pos[1] - workspace_limits[1][0])/heightmap_resolution), htmap_h-1), 0))
                ini_aperture = 0.015
                break
        
        # Monitoring
        print('pos: ', pos, ' yaw: ', yaw, ' aperture: ', ini_aperture)
        '''
        copy_color_heightmap = color_heightmap.copy()
        cv2.circle(copy_color_heightmap, (pix_x,pix_y), 5, (255, 0, 0), 3)
        pix_x_prime = int(max(min(round((pos[0]-0.02*sin(yaw) - workspace_limits[0][0])/heightmap_resolution), htmap_w-1), 0))
        pix_y_prime = int(max(min(round((pos[1]-0.02*cos(yaw) - workspace_limits[1][0])/heightmap_resolution), htmap_h-1), 0))
        cv2.arrowedLine(copy_color_heightmap, (pix_x_prime, pix_y_prime), (pix_x, pix_y), (0, 0, 255), 2)
        test_image = cv2.flip(copy_color_heightmap, 1)
        #cv2.imshow("visualization",test_image*255)
        f = plt.figure(1)
        plt.imshow(test_image[:,:,[2,1,0]])
        plt.show()
        '''
        grasp_success = robot.exe_scoop(pos, yaw, ini_aperture)

        # Save Data
        data = []
        data.append([pos[0], pos[1], pos[2], yaw, ini_aperture, grasp_success])
        cv2.imwrite('./data_collection/color_heightmap/'+str(data_order)+'_color_heightmap.jpg', color_heightmap)
        plt.imsave('./data_collection/depth_heightmap/'+str(data_order)+'_depth_heightmap.jpg', depth_heightmap)
        np.save('./data_collection/color_img/'+str(data_order)+'_color_img.npy', color_img)
        np.save('./data_collection/depth_img/'+str(data_order)+'_depth_img.npy', depth_img)
        np.save('./data_collection/color_heightmap/'+str(data_order)+'_color_heightmap.npy', color_heightmap)
        np.save('./data_collection/depth_heightmap/'+str(data_order)+'_depth_heightmap.npy', depth_heightmap)
        if not os.path.exists('./data_collection/my_data.txt'):
            np.savetxt("./data_collection/data_order.txt", [data_order], delimiter=' ')
            np.savetxt("./data_collection/my_data.txt", data, delimiter=" ")
        else:
            with open("./data_collection/data_order.txt", 'ab') as f:
                np.savetxt(f,[data_order])
            with open("./data_collection/my_data.txt", 'ab') as f:
                np.savetxt(f,data)

if __name__ == '__main__':
    #main()
    #colorizer = rs.colorizer()
    #kkk = np.load('./data_collection/depth_heightmap/6_depth_heightmap.npy')
    #cv2.imwrite('./data_collection/depth_heightmap/temp_depth_heightmap.jpg', cv2.applyColorMap(cv2.convertScaleAbs(kkk*(255/np.max(kkk)),alpha=8),cv2.COLORMAP_JET))
    #plt.imsave('./data_collection/depth_heightmap/temp_depth_heightmap.jpg', kkk*1000)