from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import torch
'''
utils
author: Liuhao Ge
'''
import torch.nn as nn
from torch.autograd import Variable
import pdb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
          self.avg = self.sum / self.count

def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None, drop=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}'.format(model_path))
    state_dict_ = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    state_dict = {}

    if drop:
        state_dict_new = {}
        flag = False
        for key in state_dict_:
            if key.startswith('T.') or key.startswith('S.'):
                flag = True
                state_dict_new[key[2:]] = state_dict_[key]
        if flag:
            state_dict_ = state_dict_new

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))

    # TODO: This remains an issue maybe.
    # for k in model_state_dict:
    #     if not (k in state_dict):
    #         print('No param {}.'.format(k))
    #         state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None, buffer_remove=False):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    state_dict_ = state_dict.copy()
    if buffer_remove:
        # remove named_buffers
        buffer_name = [x[0] for x in model.named_buffers()]
        for key in state_dict:
            if key in buffer_name:
                del state_dict_[key]

    data = {'epoch': epoch,
            'state_dict': state_dict_}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def showimg(img, nm="pic", waite=0):
  cv2.imshow(nm, img)
  return cv2.waitKey(waite)

def drawCircle(img, x, y, color=(0, 255, 0), size=2):
  for id in range(len(x)):
    cv2.circle(img, (int(x[id]), int(y[id])), 2, color, size)


def drawCirclev2(img, xy, color=(0, 255, 0), size=2):
  drawCircle(img, xy[:, 0], xy[:, 1], color, size)


def group_points(points, opt):
    # group points using knn and ball query
    # points: B * 1024 * 6
    if False: # add rgb feature
        feature_num = opt.INPUT_FEATURE_NUM + 3
    else:
        feature_num = opt.INPUT_FEATURE_NUM
    cur_train_size = len(points)
    inputs1_diff = points[:,:,0:3].transpose(1,2).unsqueeze(1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM) \
                 - points[:,0:opt.sample_num_level1,0:3].unsqueeze(-1).expand(cur_train_size,opt.sample_num_level1,3,opt.SAMPLE_NUM)# B * 512 * 3 * 1024
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 512 * 3 * 1024
    inputs1_diff = inputs1_diff.sum(2)                      # B * 512 * 1024
    dists, inputs1_idx = torch.topk(inputs1_diff, opt.knn_K, 2, largest=False, sorted=False)  # dists: B * 512 * 64; inputs1_idx: B * 512 * 64
        
    # ball query
    invalid_map = dists.gt(opt.ball_radius) # B * 512 * 64
    for jj in range(opt.sample_num_level1):
        inputs1_idx[:,jj,:][invalid_map[:,jj,:]] = jj
        
    idx_group_l1_long = inputs1_idx.view(cur_train_size,opt.sample_num_level1*opt.knn_K,1).expand(cur_train_size,opt.sample_num_level1*opt.knn_K,feature_num)
    inputs_level1 = points.gather(1,idx_group_l1_long).view(cur_train_size,opt.sample_num_level1,opt.knn_K,feature_num) # B*512*64*6

    inputs_level1_center = points[:,0:opt.sample_num_level1,0:3].unsqueeze(2)       # B*512*1*3
    # inputs_level1_imgf = emb[:,0:opt.sample_num_level1,:].unsqueeze(2)       # B*512*1*img_dim
    inputs_level1[:,:,:,0:3] = inputs_level1[:,:,:,0:3] - inputs_level1_center.expand(cur_train_size,opt.sample_num_level1,opt.knn_K,3)
    inputs_level1 = inputs_level1.unsqueeze(1).transpose(1,4).squeeze(4)  # B*6*512*64
    inputs_level1_center = inputs_level1_center.contiguous().view(-1,1,opt.sample_num_level1,3).transpose(1,3)  # B*3*512*1
    # inputs_level1_imgf = inputs_level1_imgf.contiguous().view(-1,1,opt.sample_num_level1,emb.shape[2]).transpose(1,3)  # B*img_dim*512*1
    return inputs_level1, inputs_level1_center#,inputs_level1_imgf
    #inputs_level1: B*INPUT_FEATURE_NUM*sample_num_level1*knn_K, inputs_level1_center: B*3*sample_num_level1*1, inputs_level1_imgf: B*img_dim*sample_num_level1*1

def group_points_2(points, sample_num_level1, sample_num_level2, knn_K, ball_radius):
    # group points using knn and ball query
    # points: B*(3+128)*512
    cur_train_size = points.size(0)
    inputs1_diff = points[:,0:3,:].unsqueeze(1).expand(cur_train_size,sample_num_level2,3,sample_num_level1) \
                 - points[:,0:3,0:sample_num_level2].transpose(1,2).unsqueeze(-1).expand(cur_train_size,sample_num_level2,3,sample_num_level1)# B * 128 * 3 * 512
    inputs1_diff = torch.mul(inputs1_diff, inputs1_diff)    # B * 128 * 3 * 512
    inputs1_diff = inputs1_diff.sum(2)                      # B * 128 * 512
    dists, inputs1_idx = torch.topk(inputs1_diff, knn_K, 2, largest=False, sorted=False)  # dists: B * 128 * 64; inputs1_idx: B * 128 * 64
        
    # ball query
    invalid_map = dists.gt(ball_radius) # B * 128 * 64, invalid_map.float().sum()
    #pdb.set_trace()
    for jj in range(sample_num_level2):
        inputs1_idx.data[:,jj,:][invalid_map.data[:,jj,:]] = jj

    idx_group_l1_long = inputs1_idx.view(cur_train_size,1,sample_num_level2*knn_K).expand(cur_train_size,points.size(1),sample_num_level2*knn_K)
    inputs_level2 = points.gather(2,idx_group_l1_long).view(cur_train_size,points.size(1),sample_num_level2,knn_K) # B*131*128*64

    inputs_level2_center = points[:,0:3,0:sample_num_level2].unsqueeze(3)       # B*3*128*1
    # inputs_level2_imgf = emb[:,:,0:sample_num_level2]      # B*img_dim*128*1
    inputs_level2[:,0:3,:,:] = inputs_level2[:,0:3,:,:] - inputs_level2_center.expand(cur_train_size,3,sample_num_level2,knn_K) # B*3*128*64
    return inputs_level2, inputs_level2_center#, inputs_level2_imgf
    # inputs_level2: B*131*sample_num_level2*knn_K, inputs_level2_center: B*3*sample_num_level2*1

# [[ 32.815804 , -72.75761  , 303.302    ],
#        [ 33.352417 ,  -7.1964326, 281.914    ],
#        [ 30.5873   ,  18.189898 , 264.694    ],
#        [ 28.889608 ,  36.524902 , 250.746    ],
#        [ 27.8602   ,  52.086    , 236.832    ],
#        [ 10.695703 ,  -7.4458046, 275.695    ],
#        [ 19.682495 , -15.299095 , 241.41301  ],
#        [ 23.633402 , -39.168293 , 240.87401  ],
#        [ 22.453009 , -54.2507   , 256.846    ],
#        [ -7.854335 , -14.339294 , 272.80103  ],
#        [ -1.9585693, -19.685303 , 241.91     ],
#        [  8.291778 , -37.0126   , 236.29701  ],
#        [ 16.382004 , -52.874603 , 244.95601  ],
#        [-25.676693 , -31.314201 , 267.433    ],
#        [-14.915392 , -34.978695 , 247.324    ],
#        [ -7.4376326, -46.517296 , 251.25302  ],
#        [ -7.5761337, -53.287804 , 265.084    ],
#        [ 31.957811 , -62.197308 , 274.48203  ],
#        [ 32.62351  , -46.389496 , 252.259    ],
#        [ 12.383008 , -39.5984   , 246.948    ],
#        [ -8.919479 , -34.6868   , 244.483    ]]
def handplot(points, outputs_xyz, gt_xyz, volume_length, volume_rotate, volume_offset, images):  # Visualization: show the 0th (in one batch) raw depth image, point cloud combining GT joints, point cloud combining predicted joints
    points = points.cpu().numpy()
    img_height = images.shape[1]
    img_width = images.shape[2]
    fFocal_MSRA_ = 241.42	# mm
    frm_idx = 10
    img = images[frm_idx].copy()
    xyz = gt_xyz.reshape(-1,21,3).cpu().numpy()
    volume_length = volume_length.cpu().numpy()
    volume_rotate = volume_rotate.cpu().numpy()
    volume_offset = volume_offset.cpu().numpy()
    ori_gtxyz = np.dot(((xyz[frm_idx,:,:3] + volume_offset[frm_idx])*volume_length[frm_idx]),np.linalg.inv(np.transpose(volume_rotate[frm_idx])))
    ori_points = np.dot(((points[frm_idx,:,:3] + volume_offset[frm_idx])*volume_length[frm_idx]),np.linalg.inv(np.transpose(volume_rotate[frm_idx])))
    for i in range(len(ori_points)):
        jj = img_width/2 + ori_points[i][0]*fFocal_MSRA_/ori_points[i][2] -1
        ii = img_height/2 -ori_points[i][1]*fFocal_MSRA_/ori_points[i][2] -1
        cv2.circle(img,(int(jj), int(ii)), 2, (0,0,244), 2)
        # print(ii,jj)
    cv2.imwrite('depth_jnt{}.png'.format(frm_idx),img)

def projection_batch(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d

def get_points_coordinate(depth, instrinsic_inv, device="cuda"):
    B, height, width, C = depth.size()
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),
                           torch.arange(0, width, dtype=torch.float32, device=device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)
    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
    xyz = torch.matmul(instrinsic_inv, xyz) # [B, 3, H*W]
    depth_xyz = xyz * depth.view(B, 1, -1)  # [B, 3, Ndepth, H*W]

    return depth_xyz.view(B, 3, height, width)

def get_normal(depth_np,intrinsic_np,with_normal=False):
    # load depth & intrinsic
    H, W = depth_np.shape
    depth_torch = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(-1).float() # (B, h, w, 1)
    valid_depth = depth_np > 0.0
    intrinsic_inv_np = np.linalg.inv(intrinsic_np)
    intrinsic_inv_torch = torch.from_numpy(intrinsic_inv_np).unsqueeze(0) # (B, 4, 4)
    ## step.2 compute matrix A
    # compute 3D points xyz
    points = get_points_coordinate(depth_torch, intrinsic_inv_torch[:, :3, :3], "cpu")
    if with_normal == False:
        return points.squeeze().cpu().numpy(), None
    point_matrix = F.unfold(points, kernel_size=5, stride=1, padding=4, dilation=2)

    # An = b
    matrix_a = point_matrix.view(1, 3, 25, H, W)  # (B, 3, 25, HxW)
    matrix_a = matrix_a.permute(0, 3, 4, 2, 1) # (B, HxW, 25, 3)
    matrix_a_trans = matrix_a.transpose(3, 4)
    matrix_b = torch.ones([1, H, W, 25, 1])

    # dot(A.T, A)
    point_multi = torch.matmul(matrix_a_trans, matrix_a)
    matrix_deter = torch.det(point_multi.to("cpu"))
    # make inversible
    inverse_condition = torch.ge(matrix_deter, 1e-5)
    inverse_condition = inverse_condition.unsqueeze(-1).unsqueeze(-1)
    inverse_condition_all = inverse_condition.repeat(1, 1, 1, 3, 3)
    # diag matrix to update uninverse
    diag_constant = torch.ones([3], dtype=torch.float32)
    diag_element = torch.diag(diag_constant)
    diag_element = diag_element.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    diag_matrix = diag_element.repeat(1, H, W, 1, 1)
    # inversible matrix
    inversible_matrix = torch.where(inverse_condition_all, point_multi.float(), diag_matrix)
    inv_matrix = torch.inverse(inversible_matrix.to("cpu"))

    ## step.3 compute normal vector use least square
    # n = (A.T A)^-1 A.T b // || (A.T A)^-1 A.T b ||2
    generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b)
    norm_normalize = F.normalize(generated_norm, p=2, dim=3)
    norm_normalize_np = norm_normalize.squeeze().cpu().numpy()
    
    return points.squeeze().cpu().numpy(), norm_normalize_np
    ## step.4 save normal vector
    np.save(depth_path.replace("depth", "normal"), norm_normalize_np)
    norm_normalize_draw = (((norm_normalize_np + 1) / 2) * 255).astype(np.uint8)
    cv2.imwrite(depth_path.replace("depth.npy", "normal1.png"), norm_normalize_draw)
