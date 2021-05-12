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
#from Mask_R_CNN_stone import Mask_R_CNN_stone
import random
from math import *




def main():
    robot = Robot("192.168.1.102")
    workspace_limits = robot.workspace_limits
    heightmap_resolution = robot.heightmap_resolution
    htmap_h = int(round((workspace_limits[1][1]-workspace_limits[1][0])/heightmap_resolution))
    htmap_w = int(round((workspace_limits[0][1]-workspace_limits[0][0])/heightmap_resolution))
    #mask_r_cnn_stone = Mask_R_CNN_stone()
    data_order = 200

    while(True):
        data_order += 1
        print('data order', data_order)
        while True:
            whether_shuffle_finish = input("Shuffle finish? (n or y)")
            if whether_shuffle_finish == 'y':
                break
            print('Stop shuffling!')
        color_img, depth_img = robot.getCameraData(data_order)
        color_heightmap, depth_heightmap = heightmap.get_heightmap(color_img, depth_img, robot.cam_intrinsics, robot.baseTcam, workspace_limits, heightmap_resolution)
        cv2.imwrite('./picture_20210506/color_heightmap/'+str(data_order)+'_color_heightmap.png', color_heightmap)
        cv2.imwrite('./picture_20210506/depth_heightmap/'+str(data_order)+'_depth_heightmap.png', depth_heightmap*1000)
        np.save('./picture_20210506/color_heightmap/'+str(data_order)+'_color_heightmap.npy', color_heightmap)
        np.save('./picture_20210506/depth_heightmap/'+str(data_order)+'_depth_heightmap.npy', depth_heightmap)
        if not os.path.exists('./picture_20210506/my_data.txt'):
            np.savetxt("./picture_20210506/data_order.txt", [data_order], delimiter=' ')
        else:
            with open("./picture_20210506/data_order.txt", 'ab') as f:
                np.savetxt(f,[data_order])

if __name__ == '__main__':
    main()
    #colorizer = rs.colorizer()
    #kkk = np.load('./data_collection/depth_heightmap/6_depth_heightmap.npy')
    #cv2.imwrite('./data_collection/depth_heightmap/temp_depth_heightmap.jpg', cv2.applyColorMap(cv2.convertScaleAbs(kkk*(255/np.max(kkk)),alpha=8),cv2.COLORMAP_JET))
    #plt.imsave('./data_collection/depth_heightmap/temp_depth_heightmap.jpg', kkk*1000)