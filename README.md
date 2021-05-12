# learning
This project is to use a data-driven learning-based approach to improve the performance of the scooping technique. I designed a neural network to predict a set of parameters given the RGB-D image of the bin scenario. There are in total 5 parameters of scooping: the finger contact position p, the angle between the finger direction and the vertical direction (i.e., the pitch angle) alpha, the horizontal orientation of the gripper rotated around the vertical direction (i.e., the yaw angle) beta, the gripper aperture, and the finger length differecne, as shown in the following figure.
<p align = "center">
<img src="IMG/scooping_parameters.png" width="400" height="250"> 
</p>

So far, I considered two parameters: the finger contact position p and the yaw angle beta. Other parameters are set as a fixed value. The whole data collecting process is as follows.

1. We use a Realsense SR300 camera to get the RGB image and the depth image. Then, we combine the RGB image and the depth image to make the RGB-D heightmap. A heightmap is an RGB-D image obtained from a 3D point cloud, describing the 3D information of the bin scenario. Each pixel in the heightmap is in linear relation to its horizontal position in the world frame and corresponds to a value indicating the height-from-bottom information. 
<p align = "center">
<img src="IMG/get_heightmap.png" width="400" height="250"> 
</p>

