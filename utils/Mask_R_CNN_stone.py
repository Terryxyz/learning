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
STONES_MODEL_PATH = os.path.join(ROOT_DIR, "samples/stones/mask_rcnn_stones_0100.h5")
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

    def pixel_to_camera(self, pixel, intrin, depth):
        X = (pixel[0]-intrin[0]) * depth / intrin[2]
        Y = (pixel[1]-intrin[1]) * depth / intrin[3]
        return [X, Y]

    def get_mask_pose(self, depth_intrin, depth_array, m_c, mm_pair, max_d, min_d, surf_normal):
        m_c_dist = depth_array[m_c[1], m_c[0]]/1000
        position = self.pixel_to_camera(m_c, depth_intrin, m_c_dist)
        min2max_vec = [mm_pair[0][0]-mm_pair[1][0], mm_pair[0][1]-mm_pair[1][1]]
        yaw = math.atan2(min2max_vec[1], min2max_vec[0])
        pitch = math.atan2(max_d-min_d, 0.023)
        if min2max_vec[0] > 0:
            yaw = math.pi/2 + yaw
        elif min2max_vec[0] < 0:
            if min2max_vec[1] > 0:
                yaw = -(3*math.pi/2-yaw)
            elif min2max_vec[1] < 0:
                yaw = -(-yaw-math.pi/2)
        if surf_normal == []:
            pose={'x':position[0],'y':position[1],'z':m_c_dist,'yaw':yaw,'pitch':pitch,'normal':surf_normal
            }
        else:
            pose={'x':position[0],'y':position[1],'z':m_c_dist,'yaw':yaw,'pitch':pitch,'normal':surf_normal
            }
        return pose

    def generate_stone_pose(self, depth_image, depth_array, seg_result):
        depth_copy = depth_image.copy()     
        mask_store = 4
        mask_size = []
        mask_center = []
        max_min_pair = []
        dist_min = []
        dist_max = []
        count_m = 0
        points = []
        point_show = []
        points_ero = []
        point_show_ero = []
        kernel = np.ones((5,5),np.uint8)
        image_center = [depth_copy.shape[1]/2, depth_copy.shape[0]/2]
        for m in range(seg_result['masks'].shape[2]):
    #    for m in range(seg_result['masks'].shape[2]):
            time0=time.time()
            mask = seg_result['masks'][:,:,m]
            mask = mask.astype(np.uint8)
            edges = cv2.Canny(mask,0,1)
            
            distance = []
            edge_point = []
            min_candidate = []
            max_candidate = []
            count = 0
            dist_max_ = -100000
            max_index = [0, 0]
            dist_min_ = 100000
            min_index = [0, 0]
            depth_intrin = [321.8862609863281, 238.18316650390625, 612.0938720703125, 611.785888671875] # cx, cy, fx, fy
            #print(depth_intrin)
            overall_mask_x = []
            overall_mask_y = []
            point = [] # store pointcloud of each go stone
            point_ero = [] # store pointcloud of each eroded go stone

            edge_pixel_set = np.argwhere(edges == 255)
            edge_pixel_set_copy=edge_pixel_set.tolist()
            edge_pixel_set_copy2=edge_pixel_set.tolist()
            for i in range(len(edge_pixel_set_copy2)):
                if edge_pixel_set_copy2[i][0]>=depth_copy.shape[0] or edge_pixel_set_copy2[i][1]>=depth_copy.shape[1]:
                    edge_pixel_set_copy.remove(edge_pixel_set_copy2[i])
            edge_pixel_set=np.array(edge_pixel_set_copy)
            overall_mask_x = edge_pixel_set[:,1].tolist()
            overall_mask_y = edge_pixel_set[:,0].tolist()
            for index in edge_pixel_set:
                depth_copy[index[0],index[1]] = [0, 255, 0]
            edge_point = edge_pixel_set[:]
            edge_point[:, [0, 1]] = edge_point[:, [1, 0]]
            edge_point = edge_point.tolist()
            for edge_pixel in edge_pixel_set_copy:
                if depth_array[edge_pixel[0],edge_pixel[1]]/1000 > dist_max_ and depth_array[edge_pixel[0],edge_pixel[1]]/1000 != 0:
                    dist_max_ = depth_array[edge_pixel[0],edge_pixel[1]]/1000
                    max_index = [edge_pixel[1], edge_pixel[0]] 
                if depth_array[edge_pixel[0],edge_pixel[1]]/1000 < dist_min_ and depth_array[edge_pixel[0],edge_pixel[1]]/1000 != 0:
                    dist_min_ = depth_array[edge_pixel[0],edge_pixel[1]]/1000
                    min_index = [edge_pixel[1], edge_pixel[0]] 
            #print(time.time()-time0)
                        
            pixel_radius=max(math.sqrt((max_index[0]-np.mean(overall_mask_x))**2+(max_index[1]-np.mean(overall_mask_y))**2),math.sqrt((min_index[0]-np.mean(overall_mask_x))**2+(min_index[1]-np.mean(overall_mask_y))**2))
            if True:
            #if math.sqrt((np.mean(overall_mask_x)-image_center[0])**2+(np.mean(overall_mask_y)-image_center[1])**2)<3*pixel_radius:
                erosion = cv2.erode(mask,kernel,iterations = 3)
                mask_pixel_set = np.argwhere(mask == 1)
                mask_pixel_set_copy=mask_pixel_set.tolist()
                for i in mask_pixel_set:
                    if i[0]>=depth_copy.shape[0] or i[1]>=depth_copy.shape[1]:
                        mask_pixel_set_copy.remove(i.tolist())
                for mask_pixel in mask_pixel_set_copy:
                    i = mask_pixel[0]
                    j = mask_pixel[1]
                    xy = self.pixel_to_camera([j, i], depth_intrin, depth_array[i, j])
                    point.append([xy[0], xy[1], depth_array[i, j]]) # add pointcloud
                erosion_pixel_set = np.argwhere(erosion == 1)
                erosion_pixel_set_copy=erosion_pixel_set.tolist()
                for i in erosion_pixel_set:
                    if i[0]>=depth_copy.shape[0] or i[1]>=depth_copy.shape[1]:
                        erosion_pixel_set_copy.remove(i.tolist())
                for erosion_pixel in erosion_pixel_set_copy:
                    i = erosion_pixel[0]
                    j = erosion_pixel[1]
                    xy = self.pixel_to_camera([j, i], depth_intrin, depth_array[i, j])
                    point_ero.append([xy[0], xy[1], depth_array[i, j]]) # add pointcloud
                point_show = point_show + point # pointcloud for all go stone candidates
                points.append(point)
                point_show_ero = point_show_ero + point_ero # pointcloud for all eroded go stone candidates
                points_ero.append(point_ero)
            
                #print('another dist max min', dist_max_, dist_min_)
                cv2.circle(depth_copy,tuple(edge_point[0]), 5, (0,0,255), 2)
                max_min_pair.append([max_index, min_index])
                dist_max.append(dist_max_)
                dist_min.append(dist_min_)
                mask_center.append([int(round(np.mean(overall_mask_x))), int(round(np.mean(overall_mask_y))), count_m])
                cv2.circle(depth_copy,tuple(max_min_pair[count_m][0]), 5, (0,0,0), 2)
                cv2.circle(depth_copy,tuple(max_min_pair[count_m][1]), 5, (255,0,0), 2)
                cv2.circle(depth_copy,tuple(mask_center[count_m][:2]), 5, (255,0,0), 1)
                mask_size.append(np.sum(mask))
                count_m = count_m + 1
                
        mask_index = np.argsort(-np.array(mask_size))[0:mask_store]
        
        plt.figure()
        plt.rcParams['figure.figsize'] = [24, 12]
        plt.imshow(depth_copy)
        
        surf_normal_set = dict()
        # Generate surface normal
        for k in mask_index:
            pcd_ero = o3d.geometry.PointCloud()
            pcd_ero.points = o3d.utility.Vector3dVector(points_ero[mask_center[k][2]])
            #print('points_ero', points_ero[mask_center[k][2]])
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(point_show_ero)
            o3d.io.write_point_cloud("test_ros_ero.ply", pcd_temp)
            o3d.io.write_point_cloud("test_ros_ero_selected.ply", pcd_ero)
            downpcd = o3d.geometry.voxel_down_sample(pcd_ero, voxel_size=1)
            #print('downpcd', downpcd)
            #print("Recompute the normal of the downsampled point cloud")
            o3d.geometry.estimate_normals(downpcd, search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=10, max_nn=30))
            #print("Normal as a numpy array")
        
            for i in range(np.asarray(downpcd.normals).shape[0]):
                if downpcd.normals[i][2] > 0:
                    downpcd.normals[i][0] = -downpcd.normals[i][0]
                    downpcd.normals[i][1] = -downpcd.normals[i][1]
                    downpcd.normals[i][2] = -downpcd.normals[i][2]
        
            normals = np.asarray(downpcd.normals)
            surf_normal_set[k]=np.sum(normals, axis=0) / normals.shape[0]
            #o3d.visualization.draw_geometries([downpcd])
        
        return [self.get_mask_pose(depth_intrin, depth_array, mask_center[k], max_min_pair[mask_center[k][2]], dist_max[mask_center[k][2]], dist_min[mask_center[k][2]], surf_normal_set[k]) for k in mask_index]

    def Mask_R_CNN_result(self, rgb_img_path, depth_img_path, depth_array):
        image = skimage.io.imread(rgb_img_path)
        results = self.model.detect([image], verbose=1)
        r = results[0]
        depth_image = cv2.imread(depth_img_path)
        MaskRCNNResult = self.generate_stone_pose(depth_image, depth_array, r)
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], self.class_names, r['scores'])
        return MaskRCNNResult
        

if __name__ == '__main__':
    mask_r_cnn_stone = Mask_R_CNN_stone()
    rgb_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/color_heightmap/1_color_heightmap.jpg"
    depth_img_path = "/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/depth_heightmap/1_depth_heightmap.jpg"
    depth_array = (0.2536643-(np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/sub_picture_20210422/depth_heightmap/1_depth_heightmap.npy")+0.02))*1000
    print(mask_r_cnn_stone.Mask_R_CNN_result(rgb_img_path, depth_img_path, depth_array))






