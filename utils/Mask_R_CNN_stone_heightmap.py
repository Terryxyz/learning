import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import scipy
import random
import math
import skimage.io
import datetime
import open3d as o3d
ROOT_DIR = os.path.abspath("/home/terry/Mask_RCNN/")
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
Stones_DIR = os.path.join(ROOT_DIR, "datasets/stones")
import coco
import stones_mask_R_CNN_training_setting
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
STONES_MODEL_PATH = os.path.join(ROOT_DIR, "samples/stones/mask_rcnn_stones_heightmap_0099.h5")
import time
import random
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

class InferenceConfig(stones_mask_R_CNN_training_setting.StonesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class Mask_R_CNN_stone():

    def __init__(self):
        config = InferenceConfig()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self.model.load_weights(STONES_MODEL_PATH, by_name=True)
        dataset = stones_mask_R_CNN_training_setting.StonesDataset()
        dataset.load_stones(Stones_DIR, "train")
        dataset.prepare()
        self.class_names = dataset.class_names

    def generate_stone_pose(self, rgb_image, heightmap_resolution, height_array, seg_result): #不需要除以1000
        rgb_copy = rgb_image.copy() 
        #mask_store = 4
        mask_size = []
        mask_center = []
        x_min = []
        x_max = []
        y_min = []
        y_max = []
        count_m = 0
        points = []
        point_show = []
        points_ero = []
        point_show_ero = []
        kernel = np.ones((5,5),np.uint8)
        for m in range(seg_result['masks'].shape[2]):
            time0=time.time()
            mask = seg_result['masks'][:,:,m]
            mask = mask.astype(np.uint8)
            #edges = cv2.Canny(mask,0,1)

            contours,_ = cv2.findContours(mask.copy(), 1, 2)
            cnt = contours[0]     
            for j in range(len(contours)):
                if(len(contours[j]) > len(cnt)):            
                    cnt = contours[j]            
            edge_pixel_set = cnt.reshape(-1,2)
            #edge_pixel_set[:,[1,0]]
            #print('edge_pixel_set', edge_pixel_set)
            
            count = 0
            overall_mask_x = []
            overall_mask_y = []
            point = [] # store pointcloud of each go stone
            point_ero = [] # store pointcloud of each eroded go stone

            #edge_pixel_set = np.argwhere(edges == 255)
            for index in edge_pixel_set:
                rgb_copy[index[1],index[0]] = [0, 255, 0]
            overall_mask_x = edge_pixel_set[:,0].tolist()
            overall_mask_y = edge_pixel_set[:,1].tolist()
            #print(time.time()-time0)
                        
            erosion = cv2.erode(mask,kernel,iterations = 3)
            erosion_pixel_set = np.argwhere(erosion == 1)
            erosion_pixel_set_copy=erosion_pixel_set.tolist()
            for i in erosion_pixel_set:
                if i[0]>=rgb_copy.shape[0] or i[1]>=rgb_copy.shape[1]:
                    erosion_pixel_set_copy.remove(i.tolist())
            for erosion_pixel in erosion_pixel_set_copy:
                i = erosion_pixel[0]
                j = erosion_pixel[1]
                k = height_array[i, j]/heightmap_resolution
                point_ero.append([i, j, k]) # add pointcloud
            point_show_ero = point_show_ero + point_ero # pointcloud for all eroded go stone candidates
            points_ero.append(point_ero)
        
            mask_center.append([int(round(np.mean(overall_mask_x))), int(round(np.mean(overall_mask_y))), count_m])
            x_min.append(int(round(np.min(overall_mask_x))))
            x_max.append(int(round(np.max(overall_mask_x))))
            y_min.append(int(round(np.min(overall_mask_y))))
            y_max.append(int(round(np.max(overall_mask_y))))
            cv2.circle(rgb_copy,tuple(mask_center[count_m][:2]), 5, (255,0,0), 1)
            mask_size.append(np.sum(mask))
            #print(np.sum(mask))
            count_m += 1
                
        #mask_index = np.argsort(-np.array(mask_size))[0:mask_store]
        mask_index = np.argsort(-np.array(mask_size))
        
        plt.figure()
        plt.rcParams['figure.figsize'] = [24, 12]
        plt.imshow(rgb_copy)
        
        surf_normal_set = dict()
        # Generate surface normal
        for k in mask_index:
            pcd_ero = o3d.geometry.PointCloud()
            pcd_ero.points = o3d.utility.Vector3dVector(points_ero[mask_center[k][2]])
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(point_show_ero)
            o3d.io.write_point_cloud("test_ros_ero.ply", pcd_temp)
            o3d.io.write_point_cloud("test_ros_ero_selected.ply", pcd_ero)
            downpcd = o3d.geometry.voxel_down_sample(pcd_ero, voxel_size=1)
            o3d.geometry.estimate_normals(downpcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=10, max_nn=30))
            #print("Normal as a numpy array")
        
            for i in range(np.asarray(downpcd.normals).shape[0]):
                if downpcd.normals[i][2] < 0:
                    downpcd.normals[i][0] = -downpcd.normals[i][0]
                    downpcd.normals[i][1] = -downpcd.normals[i][1]
                    downpcd.normals[i][2] = -downpcd.normals[i][2]
        
            normals = np.asarray(downpcd.normals)
            surf_normal_set[k]= np.sum(normals, axis=0) / normals.shape[0]
            surf_normal_set[k] = surf_normal_set[k] / np.linalg.norm(surf_normal_set[k])
            #o3d.visualization.draw_geometries([downpcd])
        
        return [{'x': mask_center[k][0], 'y': mask_center[k][1], 'normal': surf_normal_set[k], 'mask size': mask_size[k], 'x_min': x_min[k], 'x_max': x_max[k], 'y_min': y_min[k], 'y_max': y_max[k]} for k in mask_index]

    def Mask_R_CNN_result(self, rgb_img_path, height_array, heightmap_resolution):
        image = skimage.io.imread(rgb_img_path)
        results = self.model.detect([image], verbose=1)
        r = results[0]
        MaskRCNNResult = self.generate_stone_pose(image, heightmap_resolution, height_array, r) 
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
        return MaskRCNNResult
        

if __name__ == '__main__':
    mask_r_cnn_stone = Mask_R_CNN_stone()
    #rgb_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/color_heightmap/18_color_heightmap.jpg"
    rgb_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/picture_20210422/color_heightmap/165_color_heightmap.jpg"
    #depth_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/depth_heightmap/18_depth_heightmap.jpg"
    depth_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/picture_20210422/depth_heightmap/165_depth_heightmap.jpg"
    #height_array = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/depth_heightmap/18_depth_heightmap.npy")
    height_array = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/picture_20210422/depth_heightmap/165_depth_heightmap.npy")
    print(mask_r_cnn_stone.Mask_R_CNN_result(rgb_img_path, height_array, 0.0013/4.0))






