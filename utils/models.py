import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pcpt_res, dig_res



class DigGraspNet(nn.Module):
    def __init__(self, htmap_h, htmap_w):
        super(DigGraspNet, self).__init__()
        self.num_rotations = 8
        self.num_pitch = 3
        self.pcpt_mod = pcpt_res.ResNet(pcpt_res.BasicBlock, [1,1,1])
        self.dig_mod = dig_res.ResNet(dig_res.BasicBlock, [1,1,1], 1, htmap_h, htmap_w, self.num_pitch)

    def forward(self, input_color_data, input_depth_data, is_volatile=False, specific_rotation=-1):
        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat_list = []
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))
                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2,3,1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
                    flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())

                    # Rotate images clockwise
                    rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before, mode='nearest', align_corners=True)
                    rotate_depth = F.grid_sample(Variable(input_depth_data, volatile=True).cuda(), flow_grid_before, mode='nearest', align_corners=True)
                    rotate_color_depth = torch.cat((rotate_color, rotate_depth), dim=1)
                    #interm_feat_list.append(rotate_color) # DELETE DELETE DELETE

                    interm_feat = self.pcpt_mod(rotate_color_depth)
                    interm_feat_list.append(interm_feat)

                    # Compute sample grid for rotation AFTER perception module, undo rotation (rotate the image back)
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2,3,1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
                    flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.size())

                    output_prob.append(F.grid_sample(self.dig_mod(interm_feat), flow_grid_after, mode='nearest', align_corners=True))

                return output_prob, interm_feat_list

        else:
            self.output_prob = []
            self.interm_feat_list = []

            # Apply rotations to intermediate features
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx*(360/self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2,3,1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
            flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())

            # Rotate images clockwise
            rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)
            rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)
            rotate_color_depth = torch.cat((rotate_color, rotate_depth), dim=1)

            interm_feat = self.pcpt_mod(rotate_color_depth)
            self.interm_feat_list.append(interm_feat)

            # Compute sample grid for rotation AFTER perception module, undo rotation (rotate the image back)
            affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],[-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2,3,1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2,0,1).float()
            flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(), input_color_data.size())

            self.output_prob.append(F.grid_sample(self.dig_mod(interm_feat), flow_grid_after, mode='nearest', align_corners=True))
            return self.output_prob, self.interm_feat_list