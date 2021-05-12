import os
import numpy as np
from math import *
import math3d as m3d
import random

from arc_rotate import *
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import math3d as m3d
import logging

import matplotlib.pyplot as plt
import time
import socket
from Arduino_motor import Arduino_motor

class Robot():

    def __init__(self, tcp_host_ip):

        self.camera_width = 640
        self.camera_height = 480
        self.maxVelocity = 1.
        self.maxForce = 200.
        self.workspace_limits = np.asarray([[-0.325-0.025, -0.15+0.025], [0.65-0.025, 0.825+0.025], [0.1, 0.41]])
        self.heightmap_resolution = 0.00225

        # robot gripper parameter 
        self.finger_length = 0.125
        self.l0 = 0.0125
        self.l1 = 0.1
        self.l2l = 0.019
        self.l2r = 0.01

        logging.basicConfig(level=logging.WARN)
        self.rob = urx.Robot(tcp_host_ip) #"192.168.1.102"

        self.control_exted_thumb=Arduino_motor()
        self.tcp_host_ip = tcp_host_ip
        self.resetRobot()
        self.resetFT300Sensor()
        self.setCamera()

    def resetRobot(self):
        self.go_to_home()
        #self.gp_control(aperture)
        self.rob.set_tcp((0, 0.0, 0, 0, 0, 0))
        time.sleep(0.2)
        self.control_exted_thumb.set_thumb_length_int(180)
        self.robotiqgrip = Robotiq_Two_Finger_Gripper(self.rob)
        self.baseTee = self.rob.get_pose()
        self.baseTee.orient =  np.array([[0,  1, 0], [ 1,  0,  0], [ 0, 0, -1]])
        self.gripper_activate()
        print("reset")

    def resetFT300Sensor(self):
        HOST = self.tcp_host_ip
        PORT = 63351
        self.serialFT300Sensor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serialFT300Sensor.connect((HOST, PORT))

    def getFT300SensorData(self):
        while True:
            data = str(self.serialFT300Sensor.recv(1024),"utf-8").replace("(","").replace(")","").split(",")
            try:
                data = [float(x) for x in data]
                if len(data)==6:
                    break
            except:
                pass
        return data

    def setCamera(self):
        self.cam_intrinsics = np.asarray([[612.0938720703125, 0, 321.8862609863281], [0, 611.785888671875, 238.18316650390625], [0, 0, 1]])
        eeTcam = np.array([[0, -1, 0, 0.142],
                           [1, 0, 0, -0.003],
                           [0, 0, 1, 0.0934057+0.03],
                           [0, 0, 0, 1]])

        self.baseTcam = np.matmul(self.baseTee.get_matrix(), eeTcam)

    def getCameraData(self):
        # Set the camera settings.
        self.setCamera()
        # Setup:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, 30)
        profile = pipeline.start(config)
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print("depth_scale", depth_scale)

        # Store next frameset for later processing:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Cleanup:
        pipeline.stop()
        print("Frames Captured")

        color = np.asanyarray(color_frame.get_data())

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frames = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(aligned_depth_frame.get_data())

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        return color, depth_image*depth_scale

    def gp_control_int(self, aperture_int, delay_time = 0.2):
        self.robotiqgrip.gripper_action(aperture_int)
        time.sleep(delay_time)

    def gp_control_distance(self, aperture_distance, delay_time = 0.2): #meter
        int_list=np.array([0,20,40,60,80,100,120,130,140,150,160,170,180,185,190,195,200,205,210]) #
        distance_list= np.array([125,116.39,105.33,93.91,82.48,70.44,58.09,51.90,45.47,39.05,32.84,26.76,19.65,16.39,13.32,10.20,7.76,3.89,0])/1000
        func_from_distance_to_int = interpolate.interp1d(distance_list, int_list, kind = 3)
        self.robotiqgrip.gripper_action(int(func_from_distance_to_int(aperture_distance)))
        time.sleep(delay_time)

    def go_to_home(self):
        home_position = [126.30,-83.73,-101.82,-83.43,89.96,-53.34]
        Hong_joint0 = math.radians(home_position[0])
        Hong_joint1 = math.radians(home_position[1])
        Hong_joint2 = math.radians(home_position[2])
        Hong_joint3 = math.radians(home_position[3])
        Hong_joint4 = math.radians(home_position[4])
        Hong_joint5 = math.radians(home_position[5])

        self.rob.movej((Hong_joint0, Hong_joint1, Hong_joint2, Hong_joint3, Hong_joint4, Hong_joint5), 0.3, 0.5)

    def Frame(pos, ori):
        mat = R.from_quat(ori).as_matrix()
        F = np.concatenate(
            [np.concatenate([mat, [[0, 0, 0]]], axis=0), np.reshape([*pos, 1.], [-1, 1])], axis=1
        )
        return F

    def exe_scoop(self, pos, rot_z, ini_aperture, theta = 60*pi/180):  # rot_z rad   aperture distance
        self.go_to_home()
        if ini_aperture == 0.05:
            self.rob.set_tcp((0.0295, 0.0, 0.3389, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.04:
            self.rob.set_tcp((0.0245, 0.0, 0.3396, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.03:
            self.rob.set_tcp((0.0195, 0.0, 0.3414, 0, 0, 0))   #need change
            time.sleep(0.3)
        elif ini_aperture == 0.02:
            self.rob.set_tcp((0.0145, 0.0, 0.342, 0, 0, 0))   #need change
            time.sleep(0.3)
        else:
            raise NameError("Wrong ini aperture!")
        self.gp_control_distance(ini_aperture, delay_time = 0.2)
        self.control_exted_thumb.set_thumb_length_int(0)
        eefPose = self.rob.get_pose()
        time.sleep(0.5)
        eefPose = eefPose.get_pose_vector()
        self.rob.translate((pos[0]-eefPose[0],pos[1]-eefPose[1],0), acc=0.05, vel=0.05)
        self.rob.translate((0,0,pos[2]+0.01-eefPose[2]), acc=0.05, vel=0.05)
        self.rob.movel_tool((0,0,0,0,0,rot_z), acc=0.5, vel=0.8)
        self.rob.movel_tool((0,0,0,0,pi/2-theta,0), acc=0.5, vel=0.8, wait=True)
        self.rob.translate((0, 0, -0.02), acc=0.01, vel=0.05, wait=False)
        ini_torque_y = self.getFT300SensorData()[4]
        time0 = time.time()
        num_large_force = 0
        while True:
            if num_large_force == 3:
                self.rob.stopl()
                break
            torque_y = self.getFT300SensorData()[4]
            if torque_y > ini_torque_y+0.1:   
                num_large_force += 1
            if time.time()-time0>2.8:  
                break 
        time.sleep(0.1)

        ini_torque_y = self.getFT300SensorData()[4]
        shortest_thumb_length = self.control_exted_thumb.shortest_thumb_length
        longest_thumb_length = self.finger_length-ini_aperture*cos(60*pi/180)
        for current_thumb_length in np.arange(shortest_thumb_length,longest_thumb_length+(1e-8), (longest_thumb_length-shortest_thumb_length)/10):
            self.control_exted_thumb.set_thumb_length(current_thumb_length)     
            torque_y = self.getFT300SensorData()[4]
            if torque_y>ini_torque_y+0.2:
                break

        for aperture_distance in np.arange(ini_aperture, 0, -0.002):
            aperture_angle = self.from_aperture_distance_to_angle(aperture_distance, self.l0, self.l1, self.l2l, self.l2r)
            next_aperture_angle = self.from_aperture_distance_to_angle(aperture_distance-0.002, self.l0, self.l1, self.l2l, self.l2r)
            translate_dir_dis, extension_distance = self.scooping_parameter_finger_fixed(aperture_angle, next_aperture_angle, theta*180/pi, self.l0, self.l1, self.l2l, self.l2r, self.finger_length)
            current_thumb_length += extension_distance  
            self.control_exted_thumb.set_thumb_length(current_thumb_length)
            self.gp_control_distance(aperture_distance, delay_time = 0)
            self.rob.translate_tool((sin(theta)*translate_dir_dis[0]+cos(theta)*translate_dir_dis[1], 0, cos(theta)*translate_dir_dis[0]-sin(theta)*translate_dir_dis[1]), acc=0.1, vel=0.4, wait=False)
            #time.sleep(0.5)
        self.rob.translate_tool((0, 0, -0.12), acc=0.3, vel=0.8, wait=True)


    def finger_tip_position_wrt_gripper_frame(self, alpha, l0, l1, l2l, l2r, flength_l, flength_r):
        alpha=alpha*pi/180
        left_fingertip_position=[-l0-l1*sin(alpha)+l2l, -l1*cos(alpha)-flength_l]
        right_fingertip_position=[l0+l1*sin(alpha)-l2r, -l1*cos(alpha)-flength_r]
        return left_fingertip_position, right_fingertip_position

    def from_aperture_distance_to_angle(self, distance, l0, l1, l2l, l2r):  #angle
        return asin((distance+l2l+l2r-2*l0)/(2*l1))*180/pi

    def scooping_parameter_finger_fixed(self, current_alpha, next_alpha, theta, l0, l1, l2l, l2r, flength_r):
        theta=theta*pi/180
        current_flength_l = flength_r - (2*l0+2*l1*sin(current_alpha*pi/180)-l2l-l2r)/tan(theta)
        current_left_fingertip_pos_g, current_right_fingertip_pos_g = self.finger_tip_position_wrt_gripper_frame(current_alpha, l0, l1, l2l, l2r, current_flength_l, flength_r)  
        next_left_fingertip_pos_g, next_right_fingertip_pos_g = self.finger_tip_position_wrt_gripper_frame(next_alpha, l0, l1, l2l, l2r, current_flength_l, flength_r) 
        traslating_wst_g = [current_right_fingertip_pos_g[0]-next_right_fingertip_pos_g[0], current_right_fingertip_pos_g[1]-next_right_fingertip_pos_g[1]]
        traslating_wst_world = np.array([[sin(theta), -cos(theta)], [cos(theta), sin(theta)]]).dot(np.array([[traslating_wst_g[0]], [traslating_wst_g[1]]])).ravel().tolist()
        next_left_fingertip_pos_g = [next_left_fingertip_pos_g[0]+traslating_wst_g[0], next_left_fingertip_pos_g[1]+traslating_wst_g[1]]
        next_right_fingertip_pos_g = [next_right_fingertip_pos_g[0]+traslating_wst_g[0], next_right_fingertip_pos_g[1]+traslating_wst_g[1]]
        extension_distance = -(next_right_fingertip_pos_g[1]+(2*l0+2*l1*sin(next_alpha*pi/180)-l2l-l2r)/tan(theta)-next_left_fingertip_pos_g[1])
        return traslating_wst_world, extension_distance

if __name__ == '__main__':
    robot = Robot("192.168.1.102")
    time.sleep(1)
    #robot.gp_control(0, delay_time = 0.2)
    robot.exe_scoop([-0.24,0.72,0], -pi/2, 0.03, theta = 60*pi/180)
    # extendable most activate

