from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from sys import maxsize

import numpy as np
import torch
import torch.nn.functional as F
# from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import Textures as MeshTextures
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
    opt = self.opt
    loss = 0
    hm_loss, heatmaps_loss = 0, 0
    hand_type_loss = 0
    rotate_loss, trans_loss, pose_loss, shape_loss, joints_loss, verts_loss, verts2d_loss = 0, 0, 0, 0, 0, 0,0 
    mask_loss,dense_loss,hms_loss,wh_loss = 0,0,0,0
    if opt.reproj_loss:
      reproj_loss, norm_loss, edge_loss = 0, 0, 0
      reproj_loss_all = 0
    if opt.bone_loss:
      bone_loss, bone_dir_loss_all = 0, 0
    if opt.photometric_loss:
      norm_loss, var_loss = 0, 0
      photometric_loss, seg_loss = 0, 0
    if opt.perceptual_loss:
      perceptual_loss = 0
    if opt.gcn_decoder:
      S_loss, gcn_reproj_loss = 0, 0
    if opt.off:
      off_hm_loss, off_lms_loss, wh_loss = 0, 0, 0
    if opt.discrepancy:
      discrepancy_loss = 0
    
    if False:
      B= batch['joints_left_gt'].size(0)
      ret_rendered, ret_gpu_masks, verts_all_pred, verts_all_gt = None, None, None, None
      # add mask loss 
      mask_loss = self.smoothL1Loss(otherInfo['mask'], batch['mask'])
      # add dense loss
      dense_loss_r = self.smoothL1Loss(otherInfo['dense'][:, :3] * batch['mask'][:, :1], batch['dense'] * batch['mask'][:, :1])
      dense_loss_l = self.smoothL1Loss(otherInfo['dense'][:, 3:] * batch['mask'][:, 1:], batch['dense'] * batch['mask'][:, 1:])
      dense_loss = (dense_loss_l + dense_loss_r) / 2
      # add hms loss
      hms_loss = self.L2Loss(otherInfo['hms'], batch['hms'])
      # add mano_verts loss
      verts_left_gt = batch['verts_left_gt'] if 'verts_left_gt' in batch else None
      verts_right_gt = batch['verts_right_gt'] if 'verts_right_gt' in batch else None
      joints_left_gt = batch['joints_left_gt'] if 'joints_left_gt' in batch else None 
      joints_right_gt = batch['joints_right_gt'] if 'joints_right_gt' in batch else None  
      verts2d_left_gt = batch['verts2d_left_gt'] if 'verts2d_left_gt' in batch else None
      verts2d_right_gt = batch['verts2d_right_gt'] if 'verts2d_right_gt' in batch else None
 
      # get root_relative joints/verts gt
      joints_left_gt = torch.matmul(self.render.MANO_L.full_regressor, verts_left_gt)
      joints_right_gt = torch.matmul(self.render.MANO_R.full_regressor, verts_right_gt)    
      root_left_gt = joints_left_gt[:, 9:10]
      root_right_gt = joints_right_gt[:, 9:10]
      length_left_gt = torch.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
      length_right_gt = torch.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
      joints_left_gt_off = joints_left_gt - root_left_gt
      verts_left_gt_off = verts_left_gt - root_left_gt
      joints_right_gt_off = joints_right_gt - root_right_gt
      verts_right_gt_off = verts_right_gt - root_right_gt

      # get root_relative joints/verts prediction
      verts_left_pred_off = result['verts3d']['left']
      verts_right_pred_off = result['verts3d']['right']
      joints_left_pred_off = torch.matmul(self.render.MANO_L.full_regressor, verts_left_pred_off)
      joints_right_pred_off = torch.matmul(self.render.MANO_R.full_regressor, verts_right_pred_off)
      root_left_pred = joints_left_pred_off[:, 9:10]
      root_right_pred = joints_right_pred_off[:, 9:10]
      length_left_pred = torch.norm(joints_left_pred_off[:, 9] - joints_left_pred_off[:, 0], dim=-1)
      length_right_pred = torch.norm(joints_right_pred_off[:, 9] - joints_right_pred_off[:, 0], dim=-1)
      scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
      scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)
      joints_left_pred_off = (joints_left_pred_off - root_left_pred) * scale_left
      verts_left_pred_off = (verts_left_pred_off - root_left_pred) * scale_left
      joints_right_pred_off = (joints_right_pred_off - root_right_pred) * scale_right
      verts_right_pred_off = (verts_right_pred_off - root_right_pred) * scale_right

      # add verts loss
      verts2d_loss = self.L2Loss((result['verts2d']['left'] / self.opt.size_train[0] * 2 - 1),(verts2d_left_gt / self.opt.size_train[0] * 2 - 1)) + \
          self.L2Loss((result['verts2d']['right'] / self.opt.size_train[0] * 2 - 1),(verts2d_right_gt / self.opt.size_train[0] * 2 - 1))
      verts_loss = self.l1_loss(verts_left_pred_off, verts_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          self.l1_loss(verts_right_pred_off, verts_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]
      
      # add joints loss
      joints_loss = self.l1_loss(joints_left_pred_off, joints_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          self.l1_loss(joints_right_pred_off, joints_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]       
         
      param_scale_left = paramsDict['scale']['left']
      param_scale_right = paramsDict['scale']['right']
      param_trans2d_left = paramsDict['trans2d']['left']
      param_trans2d_right = paramsDict['trans2d']['right']    
      lms_left_pred = self.projection_batch(param_scale_left, param_trans2d_left, joints_left_pred_off, img_size=self.opt.size_train[0])
      lms_right_pred = self.projection_batch(param_scale_right, param_trans2d_right, joints_right_pred_off, img_size=self.opt.size_train[0])

      # add norm loss
      Faces_l = self.render.MANO_L.faces.astype(np.int32)
      Faces_r = self.render.MANO_R.faces.astype(np.int32)
      norm_loss = self.normal_loss(verts_left_pred_off, verts_left_gt_off, Faces_l) + self.normal_loss(verts_right_pred_off, verts_right_gt_off, Faces_r)
      edge_loss = self.edge_length_loss(verts_left_pred_off, verts_left_gt_off, Faces_l) + self.edge_length_loss(verts_right_pred_off, verts_right_gt_off, Faces_r)
      
      # add GCN verts loss
      verts_GCN_left = handDictList[0]['verts3d']['left']
      verts_GCN_right = handDictList[0]['verts3d']['right']
      verts2d_GCN_left = handDictList[0]['verts2d']['left']
      verts2d_GCN_right = handDictList[0]['verts2d']['right']
      v3dList_gt_left, v3dList_gt_right = [], []
      v2dList_gt_left, v2dList_gt_right = [], []
      # 252 to 1008
      verts_GCN_1008_left = otherInfo['converter_left'].vert_to_GCN(verts_left_gt_off)
      verts_GCN_1008_right = otherInfo['converter_right'].vert_to_GCN(verts_left_gt_off)
      verts2d_GCN_1008_left = otherInfo['converter_left'].vert_to_GCN(verts2d_left_gt)
      verts2d_GCN_1008_right = otherInfo['converter_right'].vert_to_GCN(verts2d_right_gt)

      for i in range(5):
          v3dList_gt_left.append(verts_GCN_1008_left)
          v2dList_gt_left.append(verts2d_GCN_1008_left)
          verts_GCN_1008_left = self.mesh_downsample(verts_GCN_1008_left)
          verts2d_GCN_1008_left = self.mesh_downsample(verts2d_GCN_1008_left)
          v3dList_gt_right.append(verts_GCN_1008_right)
          v2dList_gt_right.append(verts2d_GCN_1008_right)
          verts_GCN_1008_right = self.mesh_downsample(verts_GCN_1008_right)
          verts2d_GCN_1008_right = self.mesh_downsample(verts2d_GCN_1008_right)
      v3dList_gt_left.reverse()
      v2dList_gt_left.reverse()
      v3dList_gt_right.reverse()
      v2dList_gt_right.reverse()

      gcn_loss = self.l1_loss(verts_GCN_left,v3dList_gt_left[2]).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          self.l1_loss(verts_GCN_right,v3dList_gt_right[2]).reshape(B,-1).mean(dim=1)*batch['valid'][:,0]

      gcn_2d_loss = self.L2Loss(verts2d_GCN_left / self.opt.size_train[0] * 2 - 1,v2dList_gt_left[2] / self.opt.size_train[0] * 2 - 1) + \
          self.L2Loss(verts2d_GCN_right/ self.opt.size_train[0] * 2 - 1,v2dList_gt_right[2]/ self.opt.size_train[0] * 2 - 1)
          

      file_id = batch['file_id'].detach().cpu().numpy().astype(np.int)[0]

      if file_id % 500 == 0:
        save_img_0 = (np.squeeze(batch['image'][0])).detach().cpu().numpy().astype(np.float32)
        # cv2.imwrite('orig_img.jpg',save_img_0)
        lms_vis_left = verts2d_GCN_left[0]
        for id in range(len(lms_vis_left)):
          cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (0,0,255), 2)
        lms_vis_right = verts2d_GCN_right[0]
        for id in range(len(lms_vis_right)):
          cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (0,255,0), 2)
        cv2.imwrite('outputs/imgs/image_proj_left_{}.jpg'.format(file_id), save_img_0)
        if True:
          # # for rendering .obj
          Faces_l = self.render.MANO_L.faces.astype(np.int32)
          Faces_r = self.render.MANO_R.faces.astype(np.int32)
          vis_verts_left = result['verts3d']['left'].reshape(-1,778,3).detach().cpu().numpy()
          vis_verts_right = result['verts3d']['right'].reshape(-1,778,3).detach().cpu().numpy()
          gt_verts_left = verts_left_gt.reshape(-1,778,3).detach().cpu().numpy()
          gt_verts_right = verts_right_gt.reshape(-1,778,3).detach().cpu().numpy()
          k = 0 # which one in batch.
          if batch['valid'][0][0]==1: # left
            with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts_left[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
            with open('outputs/models/gt_hands_l{}.obj'.format(file_id), 'w') as f:
              for v in gt_verts_left[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                  
          if batch['valid'][0][1]==1: # right
            with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts_right[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))       
            with open('outputs/models/gt_hands_r{}.obj'.format(file_id), 'w') as f:
              for v in gt_verts_right[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))  
      verts_all_pred = torch.stack((verts_left_pred_off,verts_right_pred_off),1).reshape(B,-1,778,3)
      joints_all_pred = torch.stack((joints_left_pred_off,joints_right_pred_off),1).reshape(B,-1,21,3)  
      lms21_proj = torch.stack((lms_left_pred,lms_right_pred),1).reshape(B,-1,21,2)  
      verts_all_gt = torch.stack((verts_left_gt_off,verts_right_gt_off),1).reshape(B,-1,778,3)
      joints_all_gt = torch.stack((joints_left_gt_off,joints_right_gt_off),1).reshape(B,-1,21,3) 


      
      if mode == 'train':
        alpha = 0 if epoch < 50 else 1
        loss_stats = {}
        if opt.reproj_loss:
          loss += opt.reproj_weight * verts_loss * 100 
          loss_stats.update({'verts_loss': verts_loss})
          loss += opt.reproj_weight * mask_loss * 500 
          loss_stats.update({'mask_loss': mask_loss})      
          loss += opt.reproj_weight * dense_loss * 30 
          loss_stats.update({'dense_loss': dense_loss})
          loss += opt.reproj_weight * hms_loss * 100 
          loss_stats.update({'hms_loss': hms_loss})   
          loss += opt.reproj_weight * verts2d_loss *50
          loss_stats.update({'verts2d_loss': verts2d_loss})

          loss += opt.joints_weight * joints_loss * 100 
          loss_stats.update({'joints_loss': joints_loss})
          loss += opt.reproj_weight * norm_loss * 10
          loss_stats.update({'norm_loss': norm_loss})
          loss += opt.reproj_weight * edge_loss *2000 * alpha
          loss_stats.update({'edge_loss': edge_loss})  
          loss += opt.reproj_weight * gcn_loss *100
          loss_stats.update({'gcn_loss': gcn_loss})  
          loss += opt.reproj_weight * gcn_2d_loss *50
          loss_stats.update({'gcn_2d_loss': gcn_2d_loss})     
        loss_stats.update({'loss': loss})
      if mode == 'val' or mode == 'test':
        return verts_all_pred, joints_all_pred, verts_all_gt, joints_all_gt, lms21_proj
      else:
        return loss, loss_stats, ret_rendered, ret_gpu_masks
    else:    
      B= batch['joints_left_gt'].size(0)
      ret_rendered, ret_gpu_masks, verts_all_pred, verts_all_gt = None, None, None, None
      # add mask loss 
      mask_loss = self.smoothL1Loss(otherInfo['mask'], batch['mask'])
      # # add dense loss
      # dense_loss_r = self.smoothL1Loss(otherInfo['dense'][:, :3] * batch['mask'][:, :1], batch['dense'] * batch['mask'][:, :1])
      # dense_loss_l = self.smoothL1Loss(otherInfo['dense'][:, 3:] * batch['mask'][:, 1:], batch['dense'] * batch['mask'][:, 1:])
      # dense_loss = (dense_loss_l + dense_loss_r) / 2
      # # add hms loss
      hms_loss = self.L2Loss(otherInfo['hms'], batch['hms'])

      center_hm = _sigmoid(otherInfo['ret']['hm'])

      if mode == 'val' or mode == 'test':
        chms = center_hm.clone().detach()
        score = 0.5
        chms = _nms(chms, 5)
        K = 1
        topk_scores, pred_ind_left, topk_ys, topk_xs = _topk(chms[:,:1,:,:], K)  
        topk_scores, pred_ind_right, topk_ys, topk_xs = _topk(chms[:,1:,:,:], K)      
      ind_left = pred_ind_left if mode == 'val' or mode == 'test' else batch['ind'][:,:1]
      ind_right = pred_ind_right if mode == 'val' or mode == 'test' else batch['ind'][:,1:]
      # ind_left = batch['ind'][:,:1]
      # ind_right = batch['ind'][:,1:]      
      ## hm_loss
      hm_loss = hm_loss + self.crit(center_hm, batch['hm']) / opt.num_stacks
      # ## hand_type_loss
      # hand_type_pred_left = _tranpose_and_gather_feat(center_hm[:,:1,:,:], ind_left).reshape(-1,1)
      # hand_type_pred_right = _tranpose_and_gather_feat(center_hm[:,1:,:,:], ind_right).reshape(-1,1)
      # hand_type_pred = torch.stack((hand_type_pred_left,hand_type_pred_right),dim=1).reshape(-1,2)
      # hand_type_loss = hand_type_loss + get_hand_type_loss(hand_type_pred, batch['valid']) / opt.num_stacks   
      ## wh_loss  
      # center_wh = _sigmoid(otherInfo['ret']['wh'])                         
      #wh_loss = self.crit_lms(otherInfo['ret']['wh'], batch['valid'],
      #                          batch['ind'], batch['wh']) / opt.num_stacks 

      # add verts loss and joints loss    
      # generate verts_gt for H2O dataset.
      # add mano_verts loss
      verts_left_gt = batch['verts_left_gt'] if 'verts_left_gt' in batch else None
      verts_right_gt = batch['verts_right_gt'] if 'verts_right_gt' in batch else None
      joints_left_gt = batch['joints_left_gt'] if 'joints_left_gt' in batch else None 
      joints_right_gt = batch['joints_right_gt'] if 'joints_right_gt' in batch else None  
      verts2d_left_gt = batch['verts2d_left_gt'] if 'verts2d_left_gt' in batch else None
      verts2d_right_gt = batch['verts2d_right_gt'] if 'verts2d_right_gt' in batch else None
 
      verts_left_pred_off = result['verts3d']['left']
      verts_right_pred_off = result['verts3d']['right']

      root_left_gt = joints_left_gt[:, 9:10]
      root_right_gt = joints_right_gt[:, 9:10]
      length_left_gt = torch.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
      length_right_gt = torch.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
      joints_left_gt_off = joints_left_gt - root_left_gt
      verts_left_gt_off = verts_left_gt - root_left_gt if verts_left_gt is not None else None
      joints_right_gt_off = joints_right_gt - root_right_gt
      verts_right_gt_off = verts_right_gt - root_right_gt if verts_right_gt is not None else None

      # add verts loss
      if self.opt.dataset == 'H2O':
        verts2d_loss = self.L2Loss((result['verts2d']['left'] / self.opt.size_train[0] * 2 - 1),(verts2d_left_gt / self.opt.size_train[0] * 2 - 1)) + \
            self.L2Loss((result['verts2d']['right'] / self.opt.size_train[0] * 2 - 1),(verts2d_right_gt / self.opt.size_train[0] * 2 - 1))
        verts_loss = self.l1_loss(verts_left_pred_off, verts_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
            self.l1_loss(verts_right_pred_off, verts_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]
      
      # add joints loss
      joints_left_pred_off = torch.matmul(self.render.MANO_L.full_regressor, verts_left_pred_off) 
      joints_right_pred_off = torch.matmul(self.render.MANO_R.full_regressor, verts_right_pred_off)    
      joints_left_gt_off = torch.matmul(self.render.MANO_L.full_regressor, verts_left_gt_off) if verts_left_gt_off is not None else joints_left_gt_off
      joints_right_gt_off = torch.matmul(self.render.MANO_R.full_regressor, verts_right_gt_off) if verts_right_gt_off is not None else joints_right_gt_off
      joints_loss = self.l1_loss(joints_left_pred_off, joints_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          self.l1_loss(joints_right_pred_off, joints_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]     
      
      param_scale_left = paramsDict['scale']['left']
      param_scale_right = paramsDict['scale']['right']
      param_trans2d_left = paramsDict['trans2d']['left']
      param_trans2d_right = paramsDict['trans2d']['right']    
      lms_left_pred = self.projection_batch(param_scale_left, param_trans2d_left, joints_left_pred_off, img_size=self.opt.size_train[0])
      lms_right_pred = self.projection_batch(param_scale_right, param_trans2d_right, joints_right_pred_off, img_size=self.opt.size_train[0])
      # joints2d_loss = self.L2Loss(lms_left_pred/ self.opt.size_train[0] * 2 - 1,batch['lms_left_gt']/ self.opt.size_train[0] * 2 - 1) + \
      #     self.L2Loss(lms_right_pred/ self.opt.size_train[0] * 2 - 1,batch['lms_right_gt']/ self.opt.size_train[0] * 2 - 1)
      
      # add norm loss
      if self.opt.dataset == 'H2O':
        Faces_l = self.render.MANO_L.faces.astype(np.int32)
        Faces_r = self.render.MANO_R.faces.astype(np.int32)
        norm_loss = self.normal_loss(verts_left_pred_off, verts_left_gt_off, Faces_l) + self.normal_loss(verts_right_pred_off, verts_right_gt_off, Faces_r)
        edge_loss = self.edge_length_loss(verts_left_pred_off, verts_left_gt_off, Faces_l) + self.edge_length_loss(verts_right_pred_off, verts_right_gt_off, Faces_r)
      
        # add GCN verts loss
        verts_GCN_left = handDictList[0]['verts3d']['left']
        verts_GCN_right = handDictList[0]['verts3d']['right']
        verts2d_GCN_left = handDictList[0]['verts2d']['left']
        verts2d_GCN_right = handDictList[0]['verts2d']['right']
        v3dList_gt_left, v3dList_gt_right = [], []
        v2dList_gt_left, v2dList_gt_right = [], []
        # 252 to 1008
        verts_GCN_1008_left = otherInfo['converter_left'].vert_to_GCN(verts_left_gt_off)
        verts_GCN_1008_right = otherInfo['converter_right'].vert_to_GCN(verts_left_gt_off)
        verts2d_GCN_1008_left = otherInfo['converter_left'].vert_to_GCN(verts2d_left_gt)
        verts2d_GCN_1008_right = otherInfo['converter_right'].vert_to_GCN(verts2d_right_gt)

        for i in range(5):
            v3dList_gt_left.append(verts_GCN_1008_left)
            v2dList_gt_left.append(verts2d_GCN_1008_left)
            verts_GCN_1008_left = self.mesh_downsample(verts_GCN_1008_left)
            verts2d_GCN_1008_left = self.mesh_downsample(verts2d_GCN_1008_left)
            v3dList_gt_right.append(verts_GCN_1008_right)
            v2dList_gt_right.append(verts2d_GCN_1008_right)
            verts_GCN_1008_right = self.mesh_downsample(verts_GCN_1008_right)
            verts2d_GCN_1008_right = self.mesh_downsample(verts2d_GCN_1008_right)
        v3dList_gt_left.reverse()
        v2dList_gt_left.reverse()
        v3dList_gt_right.reverse()
        v2dList_gt_right.reverse()

        gcn_loss = self.l1_loss(verts_GCN_left,v3dList_gt_left[2]).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
            self.l1_loss(verts_GCN_right,v3dList_gt_right[2]).reshape(B,-1).mean(dim=1)*batch['valid'][:,0]

        gcn_2d_loss = self.L2Loss(verts2d_GCN_left / self.opt.size_train[0] * 2 - 1,v2dList_gt_left[2] / self.opt.size_train[0] * 2 - 1) + \
            self.L2Loss(verts2d_GCN_right/ self.opt.size_train[0] * 2 - 1,v2dList_gt_right[2]/ self.opt.size_train[0] * 2 - 1)
          
      ## if depth is acquireable, estimate absolute position with root_pred.
      if True:
        root_z_left = 0.4 + paramsDict['root']['left'][:,0] / 100
        root_z_right = 0.4 + paramsDict['root']['right'][:,0] / 100
        root_xy_left = paramsDict['root']['left'][:,1:] / 100
        root_xy_right = paramsDict['root']['right'][:,1:] / 100  
        # root_left_pred = torch.stack((root_xy_left[:,0], root_xy_left[:,1], root_z_left),1).unsqueeze(1)
        # root_right_pred = torch.stack((root_xy_right[:,0], root_xy_right[:,1], root_z_right),1).unsqueeze(1)
        root_left_pred = self.render.get_uv_root_3d(ind_left, root_xy_left, root_z_left, batch['K_new'])
        root_right_pred = self.render.get_uv_root_3d(ind_right, root_xy_right, root_z_right, batch['K_new'])
        joints_left_pred = joints_left_pred_off + root_left_pred if mode == 'val' or mode == 'test' else joints_left_pred_off + root_left_gt
        joints_right_pred = joints_right_pred_off + root_right_pred if mode == 'val' or mode == 'test' else joints_right_pred_off + root_right_gt
        lms_left_pred_proj = self.render.get_Landmarks_new(joints_left_pred,batch['K_new'])
        lms_right_pred_proj = self.render.get_Landmarks_new(joints_right_pred,batch['K_new'])
        joints2d_loss = self.L2Loss(lms_left_pred_proj/ self.opt.size_train[0] * 2 - 1,batch['lms_left_gt']/ self.opt.size_train[0] * 2 - 1)*batch['valid'][:,0] + \
            self.L2Loss(lms_right_pred_proj/ self.opt.size_train[0] * 2 - 1,batch['lms_right_gt']/ self.opt.size_train[0] * 2 - 1)*batch['valid'][:,1]
        verts_left_pred = verts_left_pred_off + root_left_pred #if mode == 'val' or mode == 'test' else verts_left_pred_off + root_left_gt
        verts_right_pred = verts_right_pred_off + root_right_pred #if mode == 'val' or mode == 'test' else verts_right_pred_off + root_right_gt
        verts2d_left_pred_proj = self.render.get_Landmarks_new(verts_left_pred,batch['K_new'])
        verts2d_right_pred_proj = self.render.get_Landmarks_new(verts_right_pred,batch['K_new'])   

        root_loss =  (self.l1_loss(root_left_pred, root_left_gt)).reshape(B,-1).mean(dim=1)*batch['valid'][:,0]*1000 + \
                (self.l1_loss(root_right_pred, root_right_gt)).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]*1000

        ### joints_loss
        abs_joints_loss = (self.l1_loss(joints_left_pred, joints_left_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          self.l1_loss(joints_right_pred, joints_right_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000
        if self.opt.dataset == 'H2O':
          abs_verts_loss = (self.l1_loss(verts_left_pred, verts_left_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
            self.l1_loss(verts_right_pred, verts_right_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000   \
          
      if opt.bone_loss:
        # lms21_pred = torch.cat((lms_left_pred_proj,lms_right_pred_proj))
        # lms21_gt = torch.cat((batch['lms_left_gt'],batch['lms_right_gt']))
        # tmp_mask = mask.reshape(-1,21,2)
        j2d_con = torch.ones_like(lms_left_pred_proj[:,:,0]).unsqueeze(-1)
        # maybe confidence can be used here.
        bone_direc_loss = get_bone_loss(lms_left_pred_proj, batch['lms_left_gt'], j2d_con).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
          get_bone_loss(lms_right_pred_proj, batch['lms_right_gt'], j2d_con).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]
        
      file_id = batch['file_id'].detach().cpu().numpy().astype(np.int)[0]

      if file_id % 100 == 0:
        save_img_0 = (np.squeeze(batch['image'][0])).detach().cpu().numpy().astype(np.float32)
        # cv2.imwrite('orig_img.jpg',save_img_0)
        # img_96 = cv2.resize(save_img_0,(96,96))
        cv2.imwrite('outputs/imgs/mask_left_{}.jpg'.format(file_id), otherInfo['mask'][0,1].detach().cpu().numpy()*255)
        cv2.imwrite('outputs/imgs/mask_right_{}.jpg'.format(file_id), otherInfo['mask'][0,0].detach().cpu().numpy()*255)
        # cv2.imwrite('outputs/imgs/center_left_{}.jpg'.format(file_id), center_hm[0,0].detach().cpu().numpy()*255)
        # cv2.imwrite('outputs/imgs/center_right_{}.jpg'.format(file_id), center_hm[0,1].detach().cpu().numpy()*255)
        lms_vis_left = lms_left_pred_proj[0]
        for id in range(len(lms_vis_left)):
          cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (0,0,255), 2)
        lms_vis_right = lms_right_pred_proj[0]
        for id in range(len(lms_vis_right)):
          cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (0,255,0), 2)
        lms_vis_left = batch['lms_left_gt'][0]
        for id in range(len(lms_vis_left)):
          cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (255,0,0), 2)
        lms_vis_right = batch['lms_right_gt'][0]
        for id in range(len(lms_vis_right)):
          cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (255,0,0), 2)        
        cv2.imwrite('outputs/imgs/image_proj_left_{}.jpg'.format(file_id), save_img_0)
        showHandJoints(save_img_0,lms_right_pred_proj[0].detach().cpu().numpy(),'outputs/imgs/kps_bone_pred_{}.jpg'.format(file_id))
        if True:
          # # for rendering .obj
          Faces_l = self.render.MANO_L.faces.astype(np.int32)
          Faces_r = self.render.MANO_R.faces.astype(np.int32)
          vis_verts_left = verts_left_pred.reshape(-1,778,3).detach().cpu().numpy()
          vis_verts_right = verts_right_pred.reshape(-1,778,3).detach().cpu().numpy()
          if self.opt.dataset == 'H2O':
            gt_verts_left = verts_left_gt.reshape(-1,778,3).detach().cpu().numpy()
            gt_verts_right = verts_right_gt.reshape(-1,778,3).detach().cpu().numpy()
          k = 0 # which one in batch.
          if batch['valid'][0][0]==1: # left
            with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts_left[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
            if self.opt.dataset == 'H2O':
              with open('outputs/models/gt_hands_l{}.obj'.format(file_id), 'w') as f:
                for v in gt_verts_left[k]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                  
          if batch['valid'][0][1]==1: # right
            with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
              for v in vis_verts_right[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
            if self.opt.dataset == 'H2O':    
              with open('outputs/models/gt_hands_r{}.obj'.format(file_id), 'w') as f:
                for v in gt_verts_right[k]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))  
          if file_id == 0:
            file_num = len(os.listdir('outputs/tmp_model'))//2
            with open('outputs/tmp_model/lhands_{}_{}.obj'.format(file_id,file_num), 'w') as f:
              for v in vis_verts_left[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
            with open('outputs/tmp_model/rhands_{}_{}.obj'.format(file_id,file_num), 'w') as f:
              for v in vis_verts_right[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 

      verts_all_pred = torch.cat((verts_left_pred,verts_right_pred),dim=1).reshape(B,-1,778,3)
      joints_all_pred = torch.cat((joints_left_pred,joints_right_pred),dim=1).reshape(B,-1,21,3)  
      lms21_proj = torch.cat((lms_left_pred_proj,lms_right_pred_proj),dim=1).reshape(B,-1,21,2)  
      verts_all_gt = torch.cat((verts_left_gt,verts_right_gt),dim=1).reshape(B,-1,778,3) if self.opt.dataset == 'H2O' else None
      joints_all_gt = torch.cat((joints_left_gt,joints_right_gt),dim=1).reshape(B,-1,21,3) 
      verts_all_pred_off = torch.cat((verts_left_pred_off,verts_right_pred_off),dim=1).reshape(B,-1,778,3)
      joints_all_pred_off = torch.cat((joints_left_pred_off,joints_right_pred_off),dim=1).reshape(B,-1,21,3)  
      verts_all_gt_off = torch.cat((verts_left_gt_off,verts_right_gt_off),dim=1).reshape(B,-1,778,3) if self.opt.dataset == 'H2O' else None
      joints_all_gt_off = torch.cat((joints_left_gt_off,joints_right_gt_off),dim=1).reshape(B,-1,21,3) 

      if mode == 'train':
        alpha = 0 if epoch < 20 else 1
        loss_stats = {}
        loss += opt.center_weight * hm_loss 
        loss_stats.update({'hm_loss': hm_loss})
        #loss += opt.wh_weight * wh_loss *0.1
        #loss_stats.update({'wh_loss': wh_loss})
        loss += opt.reproj_weight * root_loss 
        loss_stats.update({'root_loss': root_loss}) 
        if opt.reproj_loss:
          if self.opt.dataset == 'H2O':
            loss += opt.reproj_weight * verts_loss * 500 
            loss_stats.update({'verts_loss': verts_loss})
            loss += opt.reproj_weight * abs_verts_loss *0.1 
            loss_stats.update({'abs_verts_loss': abs_verts_loss})
            loss += opt.reproj_weight * verts2d_loss *50
            loss_stats.update({'verts2d_loss': verts2d_loss})
            loss += opt.reproj_weight * norm_loss * 10
            loss_stats.update({'norm_loss': norm_loss})
            loss += opt.reproj_weight * edge_loss *2000 * alpha
            loss_stats.update({'edge_loss': edge_loss})  
            loss += opt.reproj_weight * gcn_loss *100
            loss_stats.update({'gcn_loss': gcn_loss})  
            loss += opt.reproj_weight * gcn_2d_loss *50
            loss_stats.update({'gcn_2d_loss': gcn_2d_loss})  

          loss += opt.reproj_weight * mask_loss * 2000 
          loss_stats.update({'mask_loss': mask_loss})     
          loss += opt.reproj_weight * abs_joints_loss *0.1 
          loss_stats.update({'abs_joints_loss': abs_joints_loss})
          # loss += opt.reproj_weight * dense_loss * 30 
          # loss_stats.update({'dense_loss': dense_loss})
          loss += opt.reproj_weight * hms_loss * 2000 
          loss_stats.update({'hms_loss': hms_loss})   
          loss += opt.reproj_weight * joints2d_loss * 1000 * alpha # begin with large value, not optimize it.
          loss_stats.update({'joints2d_loss': joints2d_loss})
          loss += opt.reproj_weight * joints_loss * 500 
          loss_stats.update({'joints_loss': joints_loss})
 
          if opt.bone_loss:
            loss += opt.bone_dir_weight * bone_direc_loss 
            loss_stats.update({'bone_direc_loss': bone_direc_loss}) 
        loss_stats.update({'loss': loss})
      
      if mode == 'val' or mode == 'test':
        return verts_all_pred, joints_all_pred, verts_all_gt, joints_all_gt, lms21_proj, verts_all_pred_off, joints_all_pred_off, verts_all_gt_off, joints_all_gt_off
      else:
        return loss, loss_stats, ret_rendered, ret_gpu_masks

  def origforward(self, output, mode, batch, epoch):
    opt = self.opt
    hm_loss, heatmaps_loss = 0, 0
    hand_type_loss = 0
    if opt.reproj_loss:
      reproj_loss, norm_loss = 0, 0
      reproj_loss_all = 0
    if opt.bone_loss:
      bone_loss, bone_dir_loss_all = 0, 0
    if opt.photometric_loss:
      norm_loss, var_loss = 0, 0
      photometric_loss, seg_loss = 0, 0
    if opt.perceptual_loss:
      perceptual_loss = 0
    if opt.gcn_decoder:
      S_loss, gcn_reproj_loss = 0, 0
    if opt.off:
      off_hm_loss, off_lms_loss, wh_loss = 0, 0, 0
    if opt.discrepancy:
      discrepancy_loss = 0
    loss = 0
    
    for s in range(opt.num_stacks):
      # output = outputs[s]
      output['hm'] = _sigmoid(output['hm'])

      if mode == 'val' or mode == 'test':
        hms = output['hm'].clone().detach()
        score = 0.5
        hms = _nms(hms, 5)
        K = int((hms[0] > score).float().sum())
        K = 1
        topk_scores, pred_ind_left, topk_ys, topk_xs = _topk(hms[:,:1,:,:], K)  
        topk_scores, pred_ind_right, topk_ys, topk_xs = _topk(hms[:,1:,:,:], K)      
      ind_left = pred_ind_left if mode == 'val' or mode == 'test' else batch['ind'][:,:1]
      ind_right = pred_ind_right if mode == 'val' or mode == 'test' else batch['ind'][:,1:]
      ind_left = batch['ind'][:,:1]
      ind_right = batch['ind'][:,1:]      
      ## hm_loss
      hm_loss = hm_loss + self.crit(output['hm'], batch['hm']) / opt.num_stacks
      ## hand_type_loss
      hand_type_pred_left = _tranpose_and_gather_feat(output['hm'][:,:1,:,:], ind_left).reshape(-1,1)
      hand_type_pred_right = _tranpose_and_gather_feat(output['hm'][:,1:,:,:], ind_right).reshape(-1,1)
      hand_type_pred = torch.stack((hand_type_pred_left,hand_type_pred_right),dim=1).reshape(-1,2)
      # hand_type_loss = hand_type_loss + get_hand_type_loss(hand_type_pred, batch['valid']) / opt.num_stacks    

      ret_rendered, ret_gpu_masks = None, None
      file_id = batch['file_id'].detach().cpu().numpy().astype(np.int)[0]
  
      if not opt.center_only:
        if opt.off:
          ## off_hm_loss
          off_hm_loss += self.crit_lms(output['off_hm'], batch['valid'],
                                    batch['ind'], batch['off_hm']) / opt.num_stacks
          ## off_lms_loss                          
          off_lms_loss += self.crit_lms(output['off_lms'], batch['valid'],
                                    batch['ind'], batch['off_lms']) / opt.num_stacks
          ## wh_loss                          
          wh_loss += self.crit_lms(output['wh'], batch['valid'],
                                    batch['ind'], batch['wh']) / opt.num_stacks                                                                   
        if opt.reproj_loss:
          # params = _tranpose_and_gather_feat(output['params'][-1], batch['ind'])
          # params_left = _tranpose_and_gather_feat(output['params'], ind_left)
          # params_right = _tranpose_and_gather_feat(output['params'], ind_right)
          if True:
            params_left = output['point2mano_left']
            params_right = output['point2mano_right']
          else:
            params_left = output['cnn2mano_left']
            params_right = output['cnn2mano_right']  
          # off_hm_pred_left = _tranpose_and_gather_feat(output['off_hm'], ind_left) if mode == 'val' or mode == 'test' else batch['off_hm'][:,0,:]
          # off_hm_pred_right = _tranpose_and_gather_feat(output['off_hm'], ind_right) if mode == 'val' or mode == 'test' else batch['off_hm'][:,1,:]
          B, C = params_right.size(0), params_right.size(1)
          pred_orient_left, pred_pose_left, pred_shape_left, pred_trans_left, _, _, _, _  = \
            self.render.Split_coeff(params_left.view(-1, params_right.size(2)), ind_left.view(-1),batch['K_new'])
          _, _, _, _, pred_orient_right, pred_pose_right, pred_shape_right, pred_trans_right  = \
            self.render.Split_coeff(params_right.view(-1, params_right.size(2)), ind_right.view(-1),batch['K_new'])            
          # each center estimate both left/right hand params, and choose one for training.
          hand_verts_l, hand_joints_l = self.render.mano_layer_left(pred_orient_left, pred_pose_left, pred_shape_left, side ='left')
          hand_verts_r, hand_joints_r = self.render.mano_layer_right(pred_orient_right, pred_pose_right, pred_shape_right, side ='right')    
          # hand_verts_r, hand_joints_r, full_pose_r = self.render.Shape_formation(pred_orient_right, pred_pose_right, pred_shape_right, pred_trans_right,'right')              
          # hand_verts_l, hand_joints_l, full_pose_l = self.render.Shape_formation(pred_orient_left, pred_pose_left, pred_shape_left, pred_trans_left,'left')
          verts_all_pred = torch.stack((hand_verts_l,hand_verts_r),dim=1)
          joints_all_pred = torch.stack((hand_joints_l,hand_joints_r),dim=1)

          # add mano_verts loss
          verts_left_gt = batch['verts_left_gt'] if 'verts_left_gt' in batch else None
          verts_right_gt = batch['verts_right_gt'] if 'verts_right_gt' in batch else None
          joints_left_gt = batch['joints_left_gt'] if 'joints_left_gt' in batch else None 
          joints_right_gt = batch['joints_right_gt'] if 'joints_right_gt' in batch else None  
          verts2d_left_gt = batch['verts2d_left_gt'] if 'verts2d_left_gt' in batch else None
          verts2d_right_gt = batch['verts2d_right_gt'] if 'verts2d_right_gt' in batch else None
    
          # verts_all_gt = torch.stack((verts_left_gt,verts_right_gt),dim=1)
          # joints_all_gt = torch.stack((joints_left_gt,joints_right_gt),dim=1)

          # add verts_off
          root_left_gt = joints_left_gt[:, 9:10]
          root_right_gt = joints_right_gt[:, 9:10]
          length_left_gt = torch.norm(joints_left_gt[:, 9] - joints_left_gt[:, 0], dim=-1)
          length_right_gt = torch.norm(joints_right_gt[:, 9] - joints_right_gt[:, 0], dim=-1)
          joints_left_gt_off = joints_left_gt - root_left_gt
          verts_left_gt_off = verts_left_gt - root_left_gt if verts_left_gt is not None else None
          joints_right_gt_off = joints_right_gt - root_right_gt
          verts_right_gt_off = verts_right_gt - root_right_gt if verts_right_gt is not None else None

          root_left_pred = hand_joints_l[:, 9:10]
          root_right_pred = hand_joints_r[:, 9:10]
          joints_left_pred_off = hand_joints_l - root_left_pred #if mode == 'val' or mode == 'test' else joints_left_pred_off + root_left_gt
          joints_right_pred_off = hand_joints_r - root_right_pred #if mode == 'val' or mode == 'test' else joints_right_pred_off + root_right_gt
          verts_left_pred_off = hand_verts_l - root_left_pred #if mode == 'val' or mode == 'test' else verts_left_pred_off + root_left_gt
          verts_right_pred_off = hand_verts_r - root_right_pred #if mode == 'val' or mode == 'test' else verts_right_pred_off + root_right_gt
          
          # parameters for norm only, which is define from flat_hand_mean other than hand_mean.
          norm_loss = norm_loss + self.crit_norm(pred_pose_left, pred_pose_right, pred_shape_left, pred_shape_right) / opt.num_stacks

          if 'lms' in batch:
            if self.opt.dataset == 'RHD':
              hand_joints_l = joints_left_pred_off + root_left_gt
              hand_joints_r = joints_right_pred_off + root_right_gt
            lms21_proj_l = self.render.get_Landmarks_new(hand_joints_l,batch['K_new'])
            lms21_proj_r = self.render.get_Landmarks_new(hand_joints_r,batch['K_new'])  
            lms21_proj = torch.stack((lms21_proj_l,lms21_proj_r),dim=1)
            lms21_gt = torch.stack((batch['lms_left_gt'],batch['lms_right_gt']),dim=1)
            mask_l = batch['valid'][:,0].unsqueeze(1).unsqueeze(2).expand_as(lms21_proj_l).float()
            mask_r = batch['valid'][:,1].unsqueeze(1).unsqueeze(2).expand_as(lms21_proj_l).float()
         
            # get rid of outside range and not in 'skin_mask'.
            weighted_lms_mask = self.render.weighted_lms.expand_as(lms21_proj_l)

            # we consider: the larger may have larger pixel loss while they are the same estimated pose
            # center_scale = (batch['box'][:,:,2]-batch['box'][:,:,0])*(batch['box'][:,:,3]-batch['box'][:,:,1])+1e-8
            # idx1 = torch.where(center_scale<1e-3)
            # center_scale[idx1] = 3e-3 # avoid of /0
            # center_scale = center_scale * opt.input_res
            ### lms_loss
            reproj_loss_l_up = (F.mse_loss(lms21_proj_l * mask_l, batch['lms_left_gt'] * mask_l, reduction='none') \
              * weighted_lms_mask * mask_l).sum(dim=2) / ((weighted_lms_mask*mask_l).sum(dim=2) + 1e-8) / opt.num_stacks  
            reproj_loss_r_up = (F.mse_loss(lms21_proj_r * mask_r, batch['lms_right_gt'] * mask_r, reduction='none') \
              * weighted_lms_mask * mask_r).sum(dim=2) / ((weighted_lms_mask*mask_r).sum(dim=2) + 1e-8) / opt.num_stacks   
            reproj_loss_all = (reproj_loss_l_up + reproj_loss_r_up).reshape(B,-1).mean(dim=1)      

            if opt.bone_loss:
              j2d_con = torch.ones_like(lms21_proj_l[:,:,0]).unsqueeze(-1)
              # maybe confidence can be used here.
              bone_direc_loss = get_bone_loss(lms21_proj_l, batch['lms_left_gt'], j2d_con).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
                get_bone_loss(lms21_proj_r, batch['lms_right_gt'], j2d_con).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]

            ### joints_loss
            # if 'joints' in batch:
            joints_loss = (self.l1_loss(joints_left_pred_off, joints_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
              self.l1_loss(joints_right_pred_off, joints_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000
            if self.opt.dataset == 'H2O':
              verts_loss = (self.l1_loss(verts_left_pred_off, verts_left_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
                self.l1_loss(verts_right_pred_off, verts_right_gt_off).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000   
              
            root_loss =  (self.l1_loss(root_left_pred, root_left_gt)).reshape(B,-1).mean(dim=1)*batch['valid'][:,0]*1000 + \
                    (self.l1_loss(root_right_pred, root_right_gt)).reshape(B,-1).mean(dim=1)*batch['valid'][:,1]*1000

            ### abs_joints_loss
            abs_joints_loss = (self.l1_loss(hand_joints_l, joints_left_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
              self.l1_loss(hand_joints_r, joints_right_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000
            if self.opt.dataset == 'H2O':
              abs_verts_loss = (self.l1_loss(hand_verts_l, verts_left_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,0] + \
                self.l1_loss(hand_verts_r, verts_right_gt).reshape(B,-1).mean(dim=1)*batch['valid'][:,1])  *1000   
              
            verts_all_pred = torch.cat((hand_verts_l,hand_verts_r),dim=1).reshape(B,-1,778,3)
            joints_all_pred = torch.cat((hand_joints_l,hand_joints_r),dim=1).reshape(B,-1,21,3)  
            lms21_proj = torch.cat((lms21_proj_l,lms21_proj_r),dim=1).reshape(B,-1,21,2)  
            verts_all_gt = torch.cat((verts_left_gt,verts_right_gt),dim=1).reshape(B,-1,778,3) if self.opt.dataset == 'H2O' else None
            joints_all_gt = torch.cat((joints_left_gt,joints_right_gt),dim=1).reshape(B,-1,21,3) 
            verts_all_pred_off = torch.cat((verts_left_pred_off,verts_right_pred_off),dim=1).reshape(B,-1,778,3)
            joints_all_pred_off = torch.cat((joints_left_pred_off,joints_right_pred_off),dim=1).reshape(B,-1,21,3)  
            verts_all_gt_off = torch.cat((verts_left_gt_off,verts_right_gt_off),dim=1).reshape(B,-1,778,3) if self.opt.dataset == 'H2O' else None
            joints_all_gt_off = torch.cat((joints_left_gt_off,joints_right_gt_off),dim=1).reshape(B,-1,21,3) 

          else:
            lms = None 

        if opt.photometric_loss:
          pre_textures = _tranpose_and_gather_feat(output['texture'], batch['ind'])
          device = pre_textures.device
          self.render.gamma = _tranpose_and_gather_feat(output['light'], batch['ind'])
          Albedo = pre_textures.view(-1,778,3) #[b, 778, 3]
          texture_mean = torch.tensor([0.45,0.35,0.3]).float().to(device)
          texture_mean = texture_mean.unsqueeze(0).unsqueeze(0).repeat(Albedo.shape[0],Albedo.shape[1],1)#[1, 778, 3]
          Albedo = Albedo + texture_mean
          Texture, lighting = self.render.Illumination(Albedo, verts_all_pred.view(-1,778,3))

          rotShape = verts_all_pred.view(-1, C, 778, 3)
          Texture = Texture.view(-1, C, 778, 3)
          nV = rotShape.size(2)
          Verts, Faces, Textures = [], [], []
          valid = []
          tag_drop = False
          tmp_mask = batch['mask']
          for i in range(len(rotShape)):
            # detach vertex to avoid influence of color
            V_ = rotShape[i][tmp_mask[i]]#.detach()
            if V_.size(0) == 0:
              # to simplify the code, drop the whole batch with any zero_hand image.
              Verts = []
              break
            valid.append(i)
            T_ = Texture[i][tmp_mask[i]]
            F_ = torch.from_numpy(self.render.MANO_R.faces.astype(np.int64)).to(device).view(1,-1)
            range_ = torch.arange(V_.size(0)).view(-1, 1) * nV
            F_ = F_.expand(V_.size(0), -1) + range_.cuda()
            Verts.append(V_.view(-1, 3))
            Textures.append(T_.view(-1, 3))
            Faces.append(F_.view(-1, 3).float())

          if len(Verts) > 0:
            # meshes = Meshes(verts=Verts, faces=Faces,
            #                 textures=TexturesVertex(verts_features=Textures))
            meshes = Meshes(verts=Verts, faces=Faces,
                            textures=MeshTextures(verts_rgb=Textures))
            rendered, gpu_masks, depth = self.render(meshes)
            rendered = torch.flip(rendered,[1])
            gpu_masks = torch.flip(gpu_masks,[1])
            depth = torch.flip(depth,[1]) 
            gpu_masks = gpu_masks.detach().float()
            ret_rendered = rendered[-B:]
            ret_gpu_masks = gpu_masks[-B:]
            skin_mask = batch['skin_mask']
            crit = nn.L1Loss()
            tmp_seg_loss = crit(gpu_masks, skin_mask[valid])        
            seg_loss += tmp_seg_loss
            masks = gpu_masks * skin_mask[valid]
            photometric_loss += (torch.norm(rendered - batch['image'][valid], p=2, dim=3) * masks).sum() / (masks.sum() + 1e-4) / opt.num_stacks 
          else:
            tag_drop = True
            photometric_loss += torch.zeros(1).sum().cuda() / opt.num_stacks

        # Reduce the loss by orders of magnitude, 1e-2
        # reproj_loss_all = reproj_loss_all * 0.1
        # if opt.mode == 'train_3d':        
        #   pose_loss = pose_loss * 1000
          # mano_loss = mano_loss * 10
          # verts_loss = verts_loss * 1000


        # show
        if file_id % 103 == 0:
          save_img_0 = (np.squeeze(batch['image'][0])).detach().cpu().numpy().astype(np.float32)
          if False:
            wh_pred = _tranpose_and_gather_feat(output['wh'], ind_left).detach().cpu().numpy()
            off_hm_pred = _tranpose_and_gather_feat(output['off_hm'], ind_left).detach().cpu().numpy()
            off_lms_pred = _tranpose_and_gather_feat(output['off_lms'], ind_left).detach().cpu().numpy()
            ind_left = ind_left.detach().cpu().numpy()
            bit = self.opt.size_train[0] // 8
            ct_left =np.array([ind_left[0][0]%bit,ind_left[0][0]//bit])
            lms_pred = (off_lms_pred[0][0].reshape(-1,2) + ct_left)*8 
            ct_left = (ct_left+off_hm_pred[0][0])*8
            gt_ct_left = [(batch['ind'][0][0]%bit+batch['off_hm'][0][0][0])*8,(batch['ind'][0][0]//bit+batch['off_hm'][0][0][1])*8]
            w,h = wh_pred[0][0]*8
            gt_w,gt_h = batch['wh'][0][0]*8
            box_pred = (ct_left[0]-w/2), (ct_left[1]-h/2), (ct_left[0]+w/2), (ct_left[1]+h/2)
            box_gt = (gt_ct_left[0]-gt_w/2), (gt_ct_left[1]-gt_h/2), (gt_ct_left[0]+gt_w/2), (gt_ct_left[1]+gt_h/2)
            iou = get_iou(box_pred,box_gt)
            cat_name = 'left'
            txt = '{}{:.1f}'.format(cat_name, iou)
            cv2.putText(save_img_0, txt, (int(box_pred[0]), int(box_pred[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), thickness=1)
            cv2.rectangle(save_img_0, (int(gt_ct_left[0]-gt_w/2), int(gt_ct_left[1]-gt_h/2)), (int(gt_ct_left[0]+gt_w/2), int(gt_ct_left[1]+gt_h/2)), (0,0,255), 1)   
            cv2.rectangle(save_img_0, (int(ct_left[0]-w/2), int(ct_left[1]-h/2)), (int(ct_left[0]+w/2), int(ct_left[1]+h/2)), (0,255,0), 1)   
            cv2.imwrite('outputs/imgs/evaluation/boxes_{}.png'.format(file_id),save_img_0)
            showHandJoints(save_img_0,lms_pred,'outputs/imgs/evaluation/off_lms_{}.jpg'.format(file_id))

            # save_hm_gt = batch['hm'][0][1].detach().cpu().numpy()*255
            # cv2.imwrite('hm_gt{}.jpg'.format(file_id),save_hm_gt)
            # save_hm = output['hm'][0][1].detach().cpu().numpy()*255
            # cv2.imwrite('hm_pred{}.jpg'.format(file_id),save_hm)

          if opt.reproj_loss:
            vis_lms = lms21_proj.reshape(B,-1,21,2)  
            if opt.photometric_loss and rendered is not None:
              render_img = rendered[0].detach().cpu().numpy()[:, :, ::-1].astype(np.float32)
              render_msk = gpu_masks[0].detach().cpu().numpy()
              cv2.imwrite('outputs/imgs/fitted_{}_{}.jpg'.format(reproj_loss_all,file_id), render_img*255+save_img_0*((1-render_msk).reshape(save_img_0.shape[0],save_img_0.shape[1],1)))
              cv2.imwrite('outputs/imgs/gpu_masks_{}.png'.format(file_id), render_msk*255)
              cv2.imwrite('outputs/imgs/rendered_{}.jpg'.format(file_id), render_img*255)
    
          lms_vis_left = lms21_proj_l[0]
          for id in range(len(lms_vis_left)):
            cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (0,0,255), 2)
          lms_vis_right = lms21_proj_r[0]
          for id in range(len(lms_vis_right)):
            cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (0,255,0), 2)
          lms_vis_left = batch['lms_left_gt'][0]
          for id in range(len(lms_vis_left)):
            cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (255,0,0), 2)
          lms_vis_right = batch['lms_right_gt'][0]
          for id in range(len(lms_vis_right)):
            cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (255,0,0), 2)        
          cv2.imwrite('outputs/imgs/image_proj_left_{}.jpg'.format(file_id), save_img_0)

          if True:
            # # for rendering .obj
            Faces_l = self.render.MANO_L.faces.astype(np.int32)
            Faces_r = self.render.MANO_R.faces.astype(np.int32)
            vis_verts = verts_all_pred[0].reshape(-1,778,3).detach().cpu().numpy()
            if opt.photometric_loss:
              colors = Textures[0].detach().cpu().numpy()
              with open('outputs/models/colord_rhands_{}.obj'.format(file_id), 'w') as f:
                for idx in range(len(vis_verts)):
                  f.write('v %f %f %f %f %f %f\n'%(vis_verts[idx][0],vis_verts[idx][1],vis_verts[idx][2],colors[idx][0],colors[idx][1],colors[idx][2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
            if batch['valid'][0][0]==1: # left
              with open('outputs/models/lhands_{}.obj'.format(file_id), 'w') as f:
                for v in vis_verts[0]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
              if verts_left_gt is not None:
                with open('outputs/models/gt_hands_l{}.obj'.format(file_id), 'w') as f:
                  for v in verts_left_gt.reshape(-1,778,3)[0]:
                    f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                  for face in Faces_l+1:
                    f.write('f %f %f %f\n'%(face[0],face[1],face[2]))                  
            if batch['valid'][0][1]==1: # right
              with open('outputs/models/rhands_{}.obj'.format(file_id), 'w') as f:
                for v in vis_verts[1]:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))   
              if verts_right_gt is not None:
                with open('outputs/models/gt_hands_r{}.obj'.format(file_id), 'w') as f:
                  for v in verts_right_gt.reshape(-1,778,3)[0]:
                    f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                  for face in Faces_r+1:
                    f.write('f %f %f %f\n'%(face[0],face[1],face[2]))      
    if mode == 'train':
      alpha = 0 if epoch <20 else 1
      loss_stats = {}
      loss += opt.center_weight * hm_loss *0
      loss_stats.update({'hm_loss': hm_loss})
      # loss += opt.handtype_weight * hand_type_loss
      # loss_stats.update({'hand_type_loss': hand_type_loss})
      # modify 'not' to center_only for continue
      if not opt.center_only:
        if opt.off:
          loss += opt.off_weight * off_hm_loss
          loss_stats.update({'off_hm_loss': off_hm_loss})
          loss += opt.off_weight * off_lms_loss
          loss_stats.update({'off_lms_loss': off_lms_loss})
          loss += opt.wh_weight * wh_loss
          loss_stats.update({'wh_loss': wh_loss})
          
        if opt.heatmaps:
          loss += opt.heatmaps_weight * heatmaps_loss
          loss_stats.update({'heatmaps_loss':heatmaps_loss})

        if opt.reproj_loss:
          loss += opt.reproj_weight * reproj_loss_all 
          loss_stats.update({'reproj_loss_all': reproj_loss_all})
          loss += opt.norm_weight * norm_loss
          loss_stats.update({'norm_loss': norm_loss})
          if opt.bone_loss:
            loss += opt.bone_dir_weight * bone_direc_loss
            loss_stats.update({'bone_direc_loss': bone_direc_loss})  
          # if 'mano_coeff' in batch:     
            # loss += opt.mano_weight * rotate_loss *100
            # loss_stats.update({'rotate_loss': rotate_loss})   
            # loss += opt.mano_weight * trans_loss *1000
            # loss_stats.update({'trans_loss': trans_loss})                           
            # loss += opt.mano_weight * pose_loss *100
            # loss_stats.update({'pose_loss': pose_loss})   
            # loss += opt.mano_weight  * shape_loss 
            # loss_stats.update({'shape_loss': shape_loss})     
          loss += opt.reproj_weight * root_loss *0 
          loss_stats.update({'root_loss': root_loss}) 
          loss += opt.reproj_weight * abs_joints_loss *0
          loss_stats.update({'abs_joints_loss': abs_joints_loss})
          loss += opt.joints_weight * joints_loss * 10
          loss_stats.update({'joints_loss': joints_loss})
          if self.opt.dataset=='H2O':
            loss += opt.joints_weight * verts_loss 
            loss_stats.update({'verts_loss': verts_loss})                 
            loss += opt.reproj_weight * abs_verts_loss *0.01 * alpha 
            loss_stats.update({'abs_verts_loss': abs_verts_loss})

        if opt.photometric_loss:
          loss += opt.photometric_weight * photometric_loss
          loss_stats.update({'photometric_loss': photometric_loss})
          loss += opt.seg_weight * seg_loss
          loss_stats.update({'seg_loss': seg_loss})   
      loss_stats.update({'loss': loss})
    if mode == 'val' or mode == 'test':
      return verts_all_pred, joints_all_pred, verts_all_gt, joints_all_gt, lms21_proj, verts_all_pred_off, joints_all_pred_off, verts_all_gt_off, joints_all_gt_off
    else:
      return loss, loss_stats, ret_rendered, ret_gpu_masks



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
        cv2.imwrite(filename, imgIn)

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
    if False: # InterHandNew
      loss_stats = ['loss']
      # loss_stats.append('reproj_loss_all')
      loss_stats.append('mask_loss')
      loss_stats.append('dense_loss')
      loss_stats.append('hms_loss')
      loss_stats.append('verts2d_loss')
      loss_stats.append('verts_loss')
      loss_stats.append('joints_loss')
      loss_stats.append('norm_loss')
      loss_stats.append('edge_loss')
      loss_stats.append('gcn_loss')
      loss_stats.append('gcn_2d_loss')
    else:
      if True: # H2O
          loss_stats = ['loss', 'hm_loss']
          #loss_stats.append('wh_loss')  
          loss_stats.append('root_loss')  
          loss_stats.append('mask_loss')
          # loss_stats.append('dense_loss')
          loss_stats.append('hms_loss')
          loss_stats.append('joints2d_loss')
          loss_stats.append('joints_loss')
          loss_stats.append('bone_direc_loss')
          loss_stats.append('abs_joints_loss')
          if self.opt.dataset == 'H2O':
            loss_stats.append('verts2d_loss')
            loss_stats.append('verts_loss')
            loss_stats.append('norm_loss')
            loss_stats.append('edge_loss')
            loss_stats.append('gcn_loss')
            loss_stats.append('gcn_2d_loss')
            loss_stats.append('abs_verts_loss')

      else: # MANO
        loss_stats = ['loss', 'hm_loss']
        # loss_stats.append('hand_type_loss')
        if not opt.center_only:
          if opt.off:
            loss_stats.append('off_hm_loss')
            loss_stats.append('off_lms_loss')
            loss_stats.append('wh_loss')          
          if opt.heatmaps:
            loss_stats.append('heatmaps_loss')
          if opt.reproj_loss:
            loss_stats.append('reproj_loss_all')
            loss_stats.append('norm_loss')
            if opt.bone_loss:
              loss_stats.append('bone_direc_loss')
              
            loss_stats.append('root_loss') 
            loss_stats.append('joints_loss')
            loss_stats.append('abs_joints_loss')
            if self.opt.dataset == 'H2O':
              loss_stats.append('verts_loss')
              loss_stats.append('abs_verts_loss')
            # loss_stats.append('rotate_loss')
            # loss_stats.append('trans_loss')
            # loss_stats.append('pose_loss')    
            # loss_stats.append('shape_loss')
          if opt.photometric_loss:
            loss_stats.append('photometric_loss')
            loss_stats.append('seg_loss')
    loss = CtdetLoss(opt, self.render, self.facenet)
    return loss_stats, loss

