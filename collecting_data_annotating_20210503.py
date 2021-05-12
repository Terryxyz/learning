import sys
sys.path.append("./utils")
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import time
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from robot import Robot
import heightmap
import math3d as m3d
from Mask_R_CNN_stone_heightmap import Mask_R_CNN_stone
import skimage.io
from math import *
import scipy
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


heightmap_resolution = 0.0013/4.0
htmap_w = 200
htmap_h = 200

def from_rotation_angle_to_index(rotation_angle): #rad, around tcp z axis
    rotation_index = round(-rotation_angle/(pi/4))
    if rotation_index<0:
        rotation_index+=8
    return rotation_index

def fromResultToGoodBadFingerContact(MaskRCNNResult):
    goodFingerContactPixel = []
    badFingerContactPixel = []
    for ObjectPoseIndex in range(len(MaskRCNNResult)):
        ObjectPose = MaskRCNNResult[ObjectPoseIndex]
        ObjectPosition = [ObjectPose['x'], ObjectPose['y']]
        NormalInWorld = ObjectPose['normal']
        alpha0 = atan2(NormalInWorld[0], NormalInWorld[1]) # is yaw
        if str(alpha0) == 'nan':
            continue
        beta = -(pi/2+atan2(-NormalInWorld[2], sqrt(NormalInWorld[1]**2+NormalInWorld[0]**2))) #tune psi
        if str(beta) == 'nan':
            continue
        if abs(NormalInWorld[2])>0.85:
            for x_delta in np.arange(-5, 5.2, 1):
                for y_delta in np.arange(-5, 5.2, 1):
                    pos = [ObjectPosition[0]+x_delta, ObjectPosition[1]+y_delta]
                    for rotation_index in range(0,8):
                        badFingerContactPixel.append([[pos[0], pos[1]], rotation_index])       # 转前坐标，分属哪张图
            alpha_set = [0, pi/4, -pi/4, pi*3/4, -pi*3/4, pi,  pi/2, -pi/2]
            for rot_z in alpha_set:
                pos0 = [ObjectPosition[0]+(0.01/heightmap_resolution)*sin(rot_z), ObjectPosition[1]+(0.01/heightmap_resolution)*cos(rot_z)]
                for x_delta in np.arange(-2, 2.2, 1):
                    for y_delta in np.arange(-2, 2.2, 1):
                        pos = [pos0[0]+x_delta, pos0[1]+y_delta]
                        goodFingerContactPixel.append([[pos[0], pos[1]], from_rotation_angle_to_index(rot_z)])
        else:
            rot_z = alpha0
            pos0 = [ObjectPosition[0]+round((0.01/heightmap_resolution)*cos(beta)*sin(rot_z)), ObjectPosition[1]+round((0.01/heightmap_resolution)*cos(beta)*cos(rot_z))]
            for x_delta in np.arange(-2, 2.2, 1):
                for y_delta in np.arange(-2, 2.2, 1):
                    pos = [pos0[0]+x_delta, pos0[1]+y_delta]
                    goodFingerContactPixel.append([[pos[0], pos[1]], from_rotation_angle_to_index(rot_z)])
            for distance_delta in np.arange(0, 0.009*cos(beta), 0.001):
                pos0 = [ObjectPosition[0]-round((distance_delta/heightmap_resolution)*sin(rot_z)), ObjectPosition[1]-round((distance_delta/heightmap_resolution)*cos(rot_z))]
                for x_delta in np.arange(-2, 2.2, 1):
                    for y_delta in np.arange(-2, 2.2, 1):
                        pos = [pos0[0]+x_delta, pos0[1]+y_delta]
                        for rotation_index in range(0,8):
                            badFingerContactPixel.append([[pos[0], pos[1]], rotation_index])
    return goodFingerContactPixel, badFingerContactPixel


def main():
    bgr_img_folder = "./picture_20210422/color_heightmap/"
    depth_img_folder = "./picture_20210422/depth_heightmap/"
    begin_index = 1
    end_index = 200
    new_index = 0
    mask_r_cnn_stone = Mask_R_CNN_stone()
    for index in range(begin_index, end_index+1):
        print('index', index)
        if index==47:
            continue
        bgr_image_path = bgr_img_folder+str(index)+"_color_heightmap.jpg"
        depth_image_array = np.load(depth_img_folder+str(index)+"_depth_heightmap.npy")
        MaskRCNNResult = mask_r_cnn_stone.Mask_R_CNN_result(bgr_image_path, depth_image_array, heightmap_resolution)
        print(MaskRCNNResult)
        bgr_image_raw = skimage.io.imread(bgr_image_path)
        bgr_image_small = dict()
        bgr_image_small[0] = bgr_image_raw[0:200, 0:200, ::-1]
        bgr_image_small[1] = bgr_image_raw[0:200, 100:300, ::-1]
        bgr_image_small[2] = bgr_image_raw[0:200, 200:400, ::-1]
        bgr_image_small[3] = bgr_image_raw[100:300, 0:200, ::-1]
        bgr_image_small[4] = bgr_image_raw[100:300, 100:300, ::-1]
        bgr_image_small[5] = bgr_image_raw[100:300, 200:400, ::-1]
        bgr_image_small[6] = bgr_image_raw[200:400, 0:200, ::-1]
        bgr_image_small[7] = bgr_image_raw[200:400, 100:300, ::-1]
        bgr_image_small[8] = bgr_image_raw[200:400, 200:400, ::-1]
        depth_image_array_small = dict()
        depth_image_array_small[0] = depth_image_array[0:200, 0:200]
        depth_image_array_small[1] = depth_image_array[0:200, 100:300]
        depth_image_array_small[2] = depth_image_array[0:200, 200:400]
        depth_image_array_small[3] = depth_image_array[100:300, 0:200]
        depth_image_array_small[4] = depth_image_array[100:300, 100:300]
        depth_image_array_small[5] = depth_image_array[100:300, 200:400]
        depth_image_array_small[6] = depth_image_array[200:400, 0:200]
        depth_image_array_small[7] = depth_image_array[200:400, 100:300]
        depth_image_array_small[8] = depth_image_array[200:400, 200:400]
        for subindex in range(9):
            new_index+=1
            mask_belongs_to_subindex = []
            for mask_index in range(len(MaskRCNNResult)):
                if MaskRCNNResult[mask_index]['mask size']<3000:
                    continue
                if subindex==0:
                    if MaskRCNNResult[mask_index]['x_min']>=0 and MaskRCNNResult[mask_index]['x_max']<200 and MaskRCNNResult[mask_index]['y_min']>=0 and MaskRCNNResult[mask_index]['y_max']<200:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==1:
                    if MaskRCNNResult[mask_index]['x_min']>=100 and MaskRCNNResult[mask_index]['x_max']<300 and MaskRCNNResult[mask_index]['y_min']>=0 and MaskRCNNResult[mask_index]['y_max']<200:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==2:
                    if MaskRCNNResult[mask_index]['x_min']>=200 and MaskRCNNResult[mask_index]['x_max']<400 and MaskRCNNResult[mask_index]['y_min']>=0 and MaskRCNNResult[mask_index]['y_max']<200:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==3:
                    if MaskRCNNResult[mask_index]['x_min']>=0 and MaskRCNNResult[mask_index]['x_max']<200 and MaskRCNNResult[mask_index]['y_min']>=100 and MaskRCNNResult[mask_index]['y_max']<300:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==4:
                    if MaskRCNNResult[mask_index]['x_min']>=100 and MaskRCNNResult[mask_index]['x_max']<300 and MaskRCNNResult[mask_index]['y_min']>=100 and MaskRCNNResult[mask_index]['y_max']<300:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==5:
                    if MaskRCNNResult[mask_index]['x_min']>=200 and MaskRCNNResult[mask_index]['x_max']<400 and MaskRCNNResult[mask_index]['y_min']>=100 and MaskRCNNResult[mask_index]['y_max']<300:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==6:
                    if MaskRCNNResult[mask_index]['x_min']>=0 and MaskRCNNResult[mask_index]['x_max']<200 and MaskRCNNResult[mask_index]['y_min']>=200 and MaskRCNNResult[mask_index]['y_max']<400:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==7:
                    if MaskRCNNResult[mask_index]['x_min']>=100 and MaskRCNNResult[mask_index]['x_max']<300 and MaskRCNNResult[mask_index]['y_min']>=200 and MaskRCNNResult[mask_index]['y_max']<400:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
                elif subindex==8:
                    if MaskRCNNResult[mask_index]['x_min']>=200 and MaskRCNNResult[mask_index]['x_max']<400 and MaskRCNNResult[mask_index]['y_min']>=200 and MaskRCNNResult[mask_index]['y_max']<400:
                        mask_belongs_to_subindex.append(MaskRCNNResult[mask_index])
            if mask_belongs_to_subindex == []:
                continue
            rgb_image_small = bgr_image_small[subindex][:,:,[2,1,0]]
            #rgb_image_small = rgb_image_small/255
            #rgb_depth_image_small_subindex = np.concatenate((rgb_image_small, (depth_image_array_small[subindex]/0.08)[:,:,np.newaxis]), axis=2)
            # need change
            rgb_image_small = Image.fromarray(rgb_image_small)
            depth_image_small = Image.fromarray(np.round((depth_image_array_small[subindex]/0.08)*255))
            for k in range(0,8):
                rgb_image_small_k = np.array(rgb_image_small.rotate(angle = k*45, fillcolor = (0,0,0)))
                #cv2.imwrite("/home/terry/catkin_ws/rgb_image_small"+str(k)+".png", rgb_image_small_k[:,:,[2,1,0]])
                depth_image_small_k = np.array(depth_image_small.rotate(angle = k*45, fillcolor = 0))
                #cv2.imwrite("/home/terry/catkin_ws/depth_image_small"+str(k)+".png", depth_image_small_k)
                rgb_depth_image_small_subindex_k = np.concatenate((rgb_image_small_k, depth_image_small_k[:,:,np.newaxis]), axis=2).astype(np.uint8)
                if 'input_data_array' not in dir():
                    input_data_array = rgb_depth_image_small_subindex_k[np.newaxis, :]
                else:
                    input_data_array = np.concatenate((input_data_array, rgb_depth_image_small_subindex_k[np.newaxis, :]), axis=0)
            annotated_result = dict()
            for k in range(0,8):
                annotated_result[k] = np.ones((200, 200))*255
            if mask_belongs_to_subindex != []:
                goodFingerContactPixel, badFingerContactPixel = fromResultToGoodBadFingerContact(mask_belongs_to_subindex)
                #print('subindex', subindex)
                #print('good', goodFingerContactPixel)
                #print('bad', badFingerContactPixel)
                for goodIndex in range(len(goodFingerContactPixel)):
                    if subindex==0:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1], goodFingerContactPixel[goodIndex][0][0]]
                    elif subindex==1:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1], goodFingerContactPixel[goodIndex][0][0]-100]
                    elif subindex==2:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1], goodFingerContactPixel[goodIndex][0][0]-200]
                    elif subindex==3:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-100, goodFingerContactPixel[goodIndex][0][0]]
                    elif subindex==4:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-100, goodFingerContactPixel[goodIndex][0][0]-100]
                    elif subindex==5:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-100, goodFingerContactPixel[goodIndex][0][0]-200]
                    elif subindex==6:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-200, goodFingerContactPixel[goodIndex][0][0]]
                    elif subindex==7:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-200, goodFingerContactPixel[goodIndex][0][0]-100]
                    elif subindex==8:
                        dotted_position = [goodFingerContactPixel[goodIndex][0][1]-200, goodFingerContactPixel[goodIndex][0][0]-200]
                    dotted_position = [int(max(min(dotted_position[0], 199), 0)), int(max(min(dotted_position[1], 199), 0))]
                    annotated_result[goodFingerContactPixel[goodIndex][1]][dotted_position[0], dotted_position[1]] = 128
                for badIndex in range(len(badFingerContactPixel)):
                    if subindex==0:
                        dotted_position = [badFingerContactPixel[badIndex][0][1], badFingerContactPixel[badIndex][0][0]]
                    elif subindex==1:
                        dotted_position = [badFingerContactPixel[badIndex][0][1], badFingerContactPixel[badIndex][0][0]-100]
                    elif subindex==2:
                        dotted_position = [badFingerContactPixel[badIndex][0][1], badFingerContactPixel[badIndex][0][0]-200]
                    elif subindex==3:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-100, badFingerContactPixel[badIndex][0][0]]
                    elif subindex==4:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-100, badFingerContactPixel[badIndex][0][0]-100]
                    elif subindex==5:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-100, badFingerContactPixel[badIndex][0][0]-200]
                    elif subindex==6:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-200, badFingerContactPixel[badIndex][0][0]]
                    elif subindex==7:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-200, badFingerContactPixel[badIndex][0][0]-100]
                    elif subindex==8:
                        dotted_position = [badFingerContactPixel[badIndex][0][1]-200, badFingerContactPixel[badIndex][0][0]-200] 
                    dotted_position = [int(max(min(dotted_position[0], 199), 0)), int(max(min(dotted_position[1], 199), 0))]  
                    annotated_result[badFingerContactPixel[badIndex][1]][dotted_position[0], dotted_position[1]] = 0
            for k in range(0,8):
                annotated_result[k] = Image.fromarray(annotated_result[k])
                annotated_result[k] = np.array(annotated_result[k].rotate(angle = k*45, fillcolor = 255)).astype(np.uint8)
                #print(annotated_result[k])
                #plt.subplot(241+k)
                #plt.imshow(annotated_result[k])
                #cv2.imwrite("/home/terry/catkin_ws/annotated_image_small"+str(k)+".png", annotated_result[k])
                if 'label_data_array' not in dir():
                    label_data_array = annotated_result[k][np.newaxis, :]
                else:
                    label_data_array = np.concatenate((label_data_array, annotated_result[k][np.newaxis, :]), axis=0)
            #plt.show()
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_20210503/input_data_array.npy", input_data_array)
    print('input_shape', input_data_array.shape)
    np.save("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_20210503/label_data_array.npy", label_data_array)
    print('label_shape', label_data_array.shape)
            
if __name__ == '__main__':
    main()

    
    
