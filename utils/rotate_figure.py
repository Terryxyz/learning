import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable



def rotate_figure(input_color_data=None, input_depth_data=None, num_rotations=8, specific_rotation=1):

    rotate_idx = specific_rotation
    rotate_theta = np.radians(rotate_idx*(360/num_rotations))

    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],[-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
    affine_mat_before.shape = (2,3,1)
    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2,0,1).float()
    if input_color_data!=None:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_color_data.size())
    else:
        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(), input_depth_data.size())

    if input_color_data != None:
        rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)
    if input_depth_data != None:
        rotate_depth = F.grid_sample(Variable(input_depth_data, requires_grad=False).cuda(), flow_grid_before, mode='nearest', align_corners=True)
    if input_color_data != None and input_depth_data != None:
        rotate_color_depth = torch.cat((rotate_color, rotate_depth), dim=1)
    if input_color_data != None and input_depth_data == None:
        return rotate_color
    elif input_color_data == None and input_depth_data != None:
        return rotate_depth
    elif input_color_data != None and input_depth_data != None:
        return rotate_color_depth