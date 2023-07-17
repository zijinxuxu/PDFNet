from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from sys import maxsize

import numpy as np
import torch
import torch.nn.functional as F
# from pytorch3d.renderer import TexturesVertex
# from pytorch3d.structures import Meshes
# from pytorch3d.renderer.mesh.textures import Textures as MeshTextures
import torch.nn as nn
from torch.utils import data
import math
import sys
import os
from lib.models.hand3d.Mano_render import ManoRender
from lib.models.losses import FocalLoss, get_bone_loss, calculate_psnr, get_hand_type_loss, get_iou
from lib.models.losses import RegL1Loss, RegWeightedL1Loss, NormLoss
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from .base_trainer import BaseTrainer
from lib.utils.utils import drawCirclev2
import copy
import cv2
from scipy.optimize import minimize
from lib.models.networks.manolayer import ManoLayer
from lib.datasets.interhand import fix_shape
from torchvision.transforms import Resize
from lib.models.networks.mano_utils import mano_two_hands_renderer

class CtdetLoss(torch.nn.Module):
  def __init__(self, opt, render=None, facenet=None):
    super(CtdetLoss, self).__init__()
    self.opt = opt
    self.crit = FocalLoss()
    self.crit_reg = RegL1Loss()
    if opt.reproj_loss:
      self.crit_reproj = RegL1Loss()
    if opt.photometric_loss or opt.reproj_loss:
      self.crit_norm = NormLoss()
    # if opt.off:
    self.crit_lms = RegWeightedL1Loss()
    self.smoothL1Loss = nn.SmoothL1Loss()
    self.L2Loss = nn.MSELoss()
    self.render = render
    self.facenet = facenet
    self.mano_path = {'left': self.render.lhm_path,
                'right': self.render.rhm_path}           
    self.mano_layer = {'right': ManoLayer(self.mano_path['right'], center_idx=None, use_pca=True),
                        'left': ManoLayer(self.mano_path['left'], center_idx=None, use_pca=True)}
    self.twohand_renderer = mano_two_hands_renderer(img_size=(384,384), device='cuda')
    fix_shape(self.mano_layer) 
    # self.J_regressor = {'left': Jr(self.mano_layer['left'].J_regressor),
    #                 'right': Jr(self.mano_layer['right'].J_regressor)}    


  def bce_loss(self, pred, gt):
    return F.binary_cross_entropy(pred, gt, reduction='none')

  def l1_loss(self, pred, gt, is_valid=None):
    loss = F.l1_loss(pred, gt, reduction='none')
    if is_valid is not None:
        loss *= is_valid
    return loss

  def normal_loss(self, pred, gt, face, is_valid=None):
      
      v1_out = pred[:, face[:, 1], :] - pred[:, face[:, 0], :]
      v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
      v2_out = pred[:, face[:, 2], :] - pred[:, face[:, 0], :]
      v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
      v3_out = pred[:, face[:, 2], :] - pred[:, face[:, 1], :]
      v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

      v1_gt = gt[:, face[:, 1], :] - gt[:, face[:, 0], :]
      v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
      v2_gt = gt[:, face[:, 2], :] - gt[:, face[:, 0], :]
      v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
      normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
      normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

      # valid_mask = valid[:, face[:, 0], :] * valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

      cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True)) #* valid_mask
      cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True)) #* valid_mask
      cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True)) #* valid_mask
      loss = torch.cat((cos1, cos2, cos3), 1)
      if is_valid is not None:
          loss *= is_valid
      return loss.mean()


  def edge_length_loss(self, pred, gt, face, is_valid=None):

      d1_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 1], :]) ** 2, 2, keepdim=True))
      d2_out = torch.sqrt(torch.sum((pred[:, face[:, 0], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))
      d3_out = torch.sqrt(torch.sum((pred[:, face[:, 1], :] - pred[:, face[:, 2], :]) ** 2, 2, keepdim=True))

      d1_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
      d2_gt = torch.sqrt(torch.sum((gt[:, face[:, 0], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
      d3_gt = torch.sqrt(torch.sum((gt[:, face[:, 1], :] - gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

      # valid_mask_1 = valid[:, face[:, 0], :] * valid[:, face[:, 1], :]
      # valid_mask_2 = valid[:, face[:, 0], :] * valid[:, face[:, 2], :]
      # valid_mask_3 = valid[:, face[:, 1], :] * valid[:, face[:, 2], :]

      diff1 = torch.abs(d1_out - d1_gt) #* valid_mask_1
      diff2 = torch.abs(d2_out - d2_gt) #* valid_mask_2
      diff3 = torch.abs(d3_out - d3_gt) #* valid_mask_3
      loss = torch.cat((diff1, diff2, diff3), 1)
      if is_valid is not None:
          loss *= is_valid
      return loss.mean()

  def mesh_downsample(self, feat, p=2):
      # feat: bs x N x f
      feat = feat.permute(0, 2, 1).contiguous()  # x = bs x f x N
      feat = nn.AvgPool1d(p)(feat)  # bs x f x N/p
      feat = feat.permute(0, 2, 1).contiguous()  # x = bs x N/p x f
      return feat

  def mesh_upsample(self, x, p=2):
      x = x.permute(0, 2, 1).contiguous()  # x = B x F x V
      x = nn.Upsample(scale_factor=p)(x)  # B x F x (V*p)
      x = x.permute(0, 2, 1).contiguous()  # x = B x (V*p) x F
      return x
  
  def soft_argmax_1d(self, heatmap1d):     
      heatmap_size = heatmap1d.shape[2]
      heatmap1d = F.softmax(heatmap1d * heatmap_size, 2)
      coord = heatmap1d * torch.arange(heatmap_size).float().cuda()
      coord = coord.sum(dim=2, keepdim=True)
      return coord

  def projection_batch(self, scale, trans2d, label3d, img_size=256):
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
  
  def forward(self, result, paramsDict, handDictList, otherInfo, batch, mode,epoch):
    pass


  def origforward(self, output, mode, batch, epoch):
    pass




def showHandJoints(imgInOrg, gtIn, filename=None):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param filename: dump image name
    :return:
    '''
    import cv2

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = sys.version_info[0]

    gtIn = np.round(gtIn).astype(np.int)

    if gtIn.shape[0]==1:
        imgIn = cv2.circle(imgIn, center=(gtIn[0][0], gtIn[0][1]), radius=3, color=joint_color_code[0],
                             thickness=-1)
    else:

        for joint_num in range(gtIn.shape[0]):

            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=1, color=joint_color, thickness=-1)
            else:
                if PYTHON_VERSION == 3:
                    joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
                else:
                    joint_color = map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num])

                cv2.circle(imgIn, center=(gtIn[joint_num][0], gtIn[joint_num][1]), radius=1, color=joint_color, thickness=-1)

        for limb_num in range(len(limbs)):

            x1 = gtIn[limbs[limb_num][0], 1]
            y1 = gtIn[limbs[limb_num][0], 0]
            x2 = gtIn[limbs[limb_num][1], 1]
            y2 = gtIn[limbs[limb_num][1], 0]
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                if PYTHON_VERSION == 3:
                    limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
                else:
                    limb_color = map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num])

                cv2.fillConvexPoly(imgIn, polygon, color=limb_color)


    if filename is not None:
        cv2.imwrite(filename, imgIn[84:-84,:,:])

    return imgIn

def _topk(scores, K):
    b, c, h, w = scores.size()
    assert c == 1
    topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int().float()
    topk_xs = (topk_inds % w).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def _nms(heat, kernel=5):
    pad = (kernel - 1) // 2
    if kernel == 2:
        hm_pad = F.pad(heat, [0, 1, 0, 1])
        hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
    else:
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep    

def align_uv(t, uv, vertex2xyz, K):
  xyz = vertex2xyz + t
  proj = np.matmul(K, xyz.T).T
  # projection_ = proj[..., :2] / ( proj[..., 2:])
  # proj = np.matmul(K, xyz.T).T
  uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
  loss = (proj - uvz)**2
  return loss.mean()


class SimplifiedTrainer(BaseTrainer):
  def __init__(self, opt, model, optimizer=None):
    self.render = ManoRender(opt) if opt.reproj_loss or opt.photometric_loss else None
    self.facenet = None
    super(SimplifiedTrainer, self).__init__(opt, model, optimizer=optimizer)
  
  def _get_losses(self, opt):
      pass


