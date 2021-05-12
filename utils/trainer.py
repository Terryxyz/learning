import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math3d as m3d
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet, pcpt_res, dig_res, heightmap
from models import DigGraspNet

from scipy import ndimage
# if gpu is to be used
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Trainer(object):
    def __init__(self, is_testing, load_snapshot, snapshot_file, htmap_h, htmap_w):
        #self.future_reward_discount = future_reward_discount
        self.htmap_h = htmap_h
        self.htmap_w = htmap_w
        self.model = DigGraspNet(self.htmap_h, self.htmap_w).cuda()

        # Initialize Huber loss
        self.criterion = torch.nn.CrossEntropyLoss().cuda() # Cross entropy

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % (snapshot_file))

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []
        self.is_exploit_log = []

    def preload(self, transitions_directory):
        self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        self.iteration = self.executed_action_log.shape[0] - 2
        self.executed_action_log = self.executed_action_log[0:self.iteration,:]
        self.executed_action_log = self.executed_action_log.tolist()
        self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        self.label_value_log = self.label_value_log[0:self.iteration]
        self.label_value_log.shape = (self.iteration,1)
        self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        self.predicted_value_log.shape = (self.iteration,1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        self.reward_value_log.shape = (self.iteration,1)
        self.reward_value_log = self.reward_value_log.tolist()

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
        input_color_image = color_heightmap.astype(float)/255
        input_depth_image = depth_heightmap
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], 1, 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        output_prob, interm_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding, maybe here we don't have padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                contact_predict = output_prob[rotate_idx].cpu().data.numpy()
            else:
                contact_predict = np.concatenate((contact_predict, output_prob[rotate_idx].cpu().data.numpy()), axis=0)
        return contact_predict, interm_feat

    # Compute labels and backpropagate
    # TODO: Change main for args used in this function
    def backprop(self, color_heightmap, depth_heightmap, best_ctct_ind, whether_success, yaw_num = 8, aperture_num = 4):
        label_contact = np.zeros((yaw_num, aperture_num,self.htmap_h,self.htmap_w))
        label_contact[best_ctct_ind[0], best_ctct_ind[1], best_ctct_ind[2], best_ctct_ind[3]] = whether_success

        self.optimizer.zero_grad()
        loss_value = 0
        contact_predict, interm_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_ctct_ind[0])
        loss1 = self.criterion(self.model.output_prob[0][0], torch.from_numpy(label_contact).int().cuda())
        print("loss1",loss1)

        loss = loss1

        print("loss",loss)
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        print('Training loss: %f' % (loss_value))
        self.optimizer.step()


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        # prediction: [num_orientations(16), num_channels(1), height, width]
        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,0,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                print("trainer test1.0", prediction_vis.shape)
                prediction_vis.shape = (predictions.shape[2], predictions.shape[3])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas

if __name__ == '__main__':
    trainer = Trainer(is_testing=False, load_snapshot=False, snapshot_file=None, htmap_h=100, htmap_w=100)
    color_heightmap = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_collection/3_color_heightmap.npy")
    depth_heightmap = np.load("/home/terry/catkin_ws/src/dg_learning_real_one_net/data_collection/3_depth_heightmap.npy")
    trainer.backprop(color_heightmap, depth_heightmap, [0,0,50,48], 1)

