from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tkinter.messagebox import NO
from torch.functional import split

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from lib.utils.image import flip, color_aug
from lib.utils.image import get_affine_transform, affine_transform, affine_transform_array
from lib.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from lib.utils.image import draw_dense_reg
from lib.utils import data_augment, data_generators
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
from PIL import Image
from scipy import stats, ndimage
import pickle
from lib.models.networks.manolayer import ManoLayer, rodrigues_batch
from lib.utils.utils import get_normal
# import time
# import open3d as o3d
# from torch_cluster import fps


def draw_lms(img, lms):
  for id_lms in lms:
    for id in range(len(id_lms)):
      cv2.circle(img, (int(id_lms[id,0]), int(id_lms[id,1])), 2, (255,0,0), 2)
  return img

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:, :2] / uv[:, -1:]

def lms2bbox(uv):
    idx = np.where((uv[:,0] > 0)&(uv[:,1] > 0)) 
    if len(idx[0])==0:
      return np.zeros((1, 4), dtype=np.float32)     
    x_min = uv[idx,0].min()
    x_max = uv[idx,0].max()
    y_min = uv[idx,1].min()
    y_max = uv[idx,1].max()

    box_w = x_max - x_min
    box_h = y_max - y_min
    # vis
    # cv2.rectangle(mask, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0,255,0), 1)
    # save_dir = os.path.join('Data_process/cache/', mask_path.split('/')[-1])
    # cv2.imwrite('Data_process/cache/mask_{}'.format(mask_path.split('/')[-1]), mask)
    bbox = np.array([[x_min, y_min, x_max, y_max]])
    return bbox

def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2)/4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel/gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))
            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map

def process_bbox(in_box, original_img_shape):
    
    # aspect ratio preserving bbox
    bbox = in_box.copy()
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = original_img_shape[1]/original_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = int(w*2.)
    bbox[3] = int(h*2.)
    bbox[0] = int(c_x - bbox[2]/2.)
    bbox[1] = int(c_y - bbox[3]/2.)

    return bbox

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""
    # depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

    # _assert_exist(depth_filename)

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)

    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale

    return dpt


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        # print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1

class InterHandDataset(data.Dataset):

  def _get_border(self, border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

  def normal(self, img):
    res = img.astype(np.float32) / 255.
    return (res - self.mean) / self.std

  def pad(self, img, stride):
    img = self.normal(img)
    height,width = img.shape[:2]
    padh = math.ceil(height / stride) * stride - height
    padw = math.ceil(width / stride) * stride - width
    result = np.pad(img, ((0,padh), (0,padw), (0,0)), mode='constant')
    assert result.shape[0] % stride == 0 and result.shape[1] % stride == 0
    return result


  def farthest_point_sampling_fast(self, point_cloud, sample_num):     
      # farthest point sampling
      # point_cloud: Nx3
      # author: Liuhao Ge
      pc_num = point_cloud.shape[0]

      if pc_num <= sample_num:
          sampled_idx = np.arange(pc_num)
          # sampled_idx = [sampled_idx; randi([1,pc_num],sample_num-pc_num,1)];
          sampled_idx = np.append(sampled_idx,np.random.randint(pc_num, size=((sample_num-pc_num),1)))
      else:
          sampled_idx = np.zeros((sample_num,),dtype=int)
          sampled_idx[0] = np.random.randint(pc_num)
          
          # cur_sample = repmat(point_cloud[sampled_idx[0],:],pc_num,1);
          cur_sample = np.tile(point_cloud[sampled_idx[0],:],(pc_num,1))
          diff = point_cloud - cur_sample
          # min_dist = sum(diff.*diff, 2);
          min_dist = np.sum(diff*diff,1)

          for cur_sample_idx in range(1,sample_num):
              # find the farthest point
              sampled_idx[cur_sample_idx]= np.argmax(min_dist)
              
              if cur_sample_idx < sample_num:
                  # update min_dist
                  valid_idx = (min_dist>1e-8)
                  diff = point_cloud[valid_idx,:] - np.tile(point_cloud[sampled_idx[cur_sample_idx],:],(sum(valid_idx),1))
                  min_dist[valid_idx] = np.minimum(min_dist[valid_idx], np.sum(diff*diff,1))
      # here may be less than 512 points. repeat will cause error.
      sampled_idx = np.unique(sampled_idx)
      return sampled_idx

  # auxiliary function
  def depth_two_uint8_to_float(self, top_bits, bottom_bits):
      """ Converts a RGB-coded depth into float valued depth. """
      depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
      depth_map /= float(2**16 - 1)
      depth_map *= 5.0
      return depth_map
      
  def __getitem__(self, index):
    mano_path = {'left': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_LEFT.pkl'),
                'right': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_RIGHT.pkl')}            
    if self.opt.dataset == 'InterHandNew':
      self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None, use_pca=True),
                          'left': ManoLayer(mano_path['left'], center_idx=None, use_pca=True)}
      fix_shape(self.mano_layer)          
      img = cv2.imread(os.path.join(self.data_path, self.split, 'img', '{}.jpg'.format(index)))
      mask = cv2.imread(os.path.join(self.data_path, self.split, 'mask', '{}.jpg'.format(index)))
      dense = cv2.imread(os.path.join(self.data_path, self.split, 'dense', '{}.jpg'.format(index)))
      bool_flip = True if np.random.randint(0, 2) == 0 else False
      # add img noise
      if self.opt.brightness and np.random.randint(0, 2) == 0:
        img = data_augment.add_noise(img.astype(np.float32),
                                      noise=0.0,
                                      scale=255.0,
                                      alpha=0.3, beta=0.05).astype(np.uint8)
      if bool_flip:
          img = cv2.flip(img, 1)
          mask = cv2.flip(mask, 1)
          dense = cv2.flip(dense, 1)
      with open(os.path.join(self.data_path, self.split, 'anno', '{}.pkl'.format(index)), 'rb') as file:
          data = pickle.load(file)

      R = data['camera']['R']
      T = data['camera']['t']
      camera = data['camera']['camera']

      hand_dict = {}
      for hand_type in ['left', 'right']:

          params = data['mano_params'][hand_type]
          handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params['R']).float(),
                                                    torch.from_numpy(params['pose']).float(),
                                                    torch.from_numpy(params['shape']).float(),
                                                    trans=torch.from_numpy(params['trans']).float())
          handV = handV[0].numpy()
          handJ = handJ[0].numpy()
          handV = handV @ R.T + T
          handJ = handJ @ R.T + T

          handV2d = handV @ camera.T
          handV2d = handV2d[:, :2] / handV2d[:, 2:]
          handJ2d = handJ @ camera.T
          handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]

          if bool_flip:
            handJ2d[:,0] = img.shape[0] - handJ2d[:, 0]
            handV2d[:,0] = img.shape[0] - handV2d[:, 0]
            handJ[:, 0] = -handJ[:,0]
            handV[:, 0] = -handV[:,0]

          hand_dict[hand_type] = {#'hms': hms,
                                  'verts3d': handV, 'joints3d': handJ,
                                  'verts2d': handV2d, 'joints2d': handJ2d,
                                  'R': R @ params['R'][0],
                                  'pose': params['pose'][0],
                                  'shape': params['shape'][0],
                                  'camera': camera
                                  }

      if True: # do augmentation operation here to get 10mm accuracy, now get 15mm accuracy.
        c = np.array([img.shape[0] / 2., img.shape[1] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.
        min_scale, max_scale = 0.9, 1.1
        s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))
        center = np.array([img.shape[0] / 2., img.shape[1] / 2.], dtype=np.float32)
        center_noise = 5
        c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
        c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))
        # max_size = max(twohand_bbox[0, 2:] - twohand_bbox[0, :2] + 1) 
        # min_scale, max_scale = max_size / 0.7 / s, max_size / 0.6 / s
        # s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))   
        rot = np.random.randint(low=-90, high=90) # not defined yet.

        trans_input,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
        img = cv2.warpAffine(img, trans_input,
                            (self.opt.size_train[0], self.opt.size_train[1]),
                            flags=cv2.INTER_LINEAR)
        # if self.opt.depth:
        #   depth = cv2.warpAffine(depth, trans_input,
        #                       (self.opt.size_train[0], self.opt.size_train[1]),
        #                       flags=cv2.INTER_LINEAR)
          # depth = np.clip(depth,0,2) / 2.0        # clip to [0,2] in meters, and normalize to [0,1]         

        if mask is not None:
          mask = cv2.warpAffine(mask, trans_input, (self.opt.size_train[0], self.opt.size_train[1]),
                                flags=cv2.INTER_NEAREST)
          # save mask to [64,64]
          mask = cv2.resize(mask,(64,64))
          ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
          mask = mask.astype(np.float) / 255
          mask = mask[..., 1:]
          # mask_left_gt = (mask[:,:,2]>100).astype(np.uint8) 
          # mask_right_gt = (mask[:,:,1]>100).astype(np.uint8)
          if bool_flip:
              mask = mask[..., [1, 0]]   
          mask = mask.transpose(2, 0, 1)
          mask_right_gt = mask[0]
          mask_left_gt = mask[1]

        if dense is not None:
          dense = cv2.warpAffine(dense, trans_input,
                              (self.opt.size_train[0], self.opt.size_train[1]),
                              flags=cv2.INTER_LINEAR)
          dense = cv2.resize(dense,(64,64)) / 255
          dense = dense.transpose(2, 0, 1) 
        # rotate lms and joints
        tx, ty = trans_input[0,2], trans_input[1,2]
        cx, cy, fx, fy= camera[0,2],camera[1,2],camera[0,0],camera[1,1]
        t0 = (trans_input[0,0] * cx + trans_input[0,1] * cy +tx -cx) / fx
        t1 = (trans_input[1,0] * cx + trans_input[1,1] * cy +ty -cy) / fy
        rot_point = np.array([[np.cos(rot / 180. * np.pi), np.sin(rot / 180. * np.pi), t0],
                [-np.sin(rot / 180. * np.pi), np.cos(rot / 180. * np.pi), t1],
                [0, 0, 1]])
        rot_point[:2,:2] = trans_input[:2,:2].copy()

        hms = np.zeros((42, 64, 64), dtype=np.float32)
        hms_idx = 0          
        for hand_type in ['left', 'right']:
          for hIdx in range(7):
            hm = cv2.imread(os.path.join(self.data_path, self.split, 'hms', '{}_{}_{}.jpg'.format(index, hIdx, hand_type)))
            hm = cv2.resize(hm,(256,256))
            hm = cv2.warpAffine(hm, trans_input,
                                (self.opt.size_train[0], self.opt.size_train[1]),
                                flags=cv2.INTER_LINEAR)  
            if bool_flip:
               hm = cv2.flip(hm, 1)          
            hm = cv2.resize(hm,(64,64))
            hm = hm.transpose(2, 0, 1) / 255.
            for kk in range(hm.shape[0]):
              hms[hms_idx] = hm[kk]
              hms_idx = hms_idx + 1

          hand_dict[hand_type]['joints2d'] = affine_transform_array(hand_dict[hand_type]['joints2d'],trans_input)
          hand_dict[hand_type]['verts2d'] = affine_transform_array(hand_dict[hand_type]['verts2d'],trans_input)

          hand_dict[hand_type]['joints3d'] = np.matmul(hand_dict[hand_type]['joints3d'],rot_point.T)
          hand_dict[hand_type]['verts3d'] = np.matmul(hand_dict[hand_type]['verts3d'],rot_point.T)

          ## proj to image for vis
          if False:
            handV2d = hand_dict[hand_type]['joints3d'] @ camera.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            lms_img = draw_lms(img, handV2d.reshape(1,-1,2))
            cv2.imwrite('local_lms_mixed_img.jpg', lms_img)
            with open('hands_{}.obj'.format(33), 'w') as f:
              for v in hand_dict[hand_type]['verts3d'].reshape(-1,778,3)[0]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))

        if bool_flip:
            idx = [i + 21 for i in range(21)] + [i for i in range(21)]
            hms = hms[idx,...]
      else:
        # save mask to [64,64] 
        mask_gt =((mask[:,:,2]>10).astype(np.uint8) | (mask[:,:,1]>10).astype(np.uint8))
        mask_gt = cv2.resize(mask_gt,(64,64))
        dense = cv2.resize(dense,(64,64))
      if bool_flip:
         lms_left = hand_dict['right']['joints2d']
         lms_right = hand_dict['left']['joints2d']
      else:
        lms_left = hand_dict['left']['joints2d']
        lms_right = hand_dict['right']['joints2d']

      bbox_left = lms2bbox(lms_left)
      ct_left = (bbox_left[0, 2:] + bbox_left[0, :2]) / 2
      left_w, left_h = (bbox_left[:, 2] - bbox_left[:, 0])/0.8, (bbox_left[:, 3] - bbox_left[:, 1])/0.8

      bbox_right = lms2bbox(lms_right)
      ct_right = (bbox_right[0, 2:] + bbox_right[0, :2]) / 2
      right_w, right_h = (bbox_right[:, 2] - bbox_right[:, 0])/0.8, (bbox_right[:, 3] - bbox_right[:, 1])/0.8

      # calculate 1/8 gts
      down = self.opt.down_ratio
      #down = 4 # test for IntagHand format
      self.max_objs = 2 
      self.num_classes = 2 # left/right hand
      heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
      hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
      hm_lms = np.zeros((42, heatmap_h, heatmap_w), dtype=np.float32)
      wh = np.zeros((self.max_objs, 2), dtype=np.float32)
      off_hm = np.zeros((self.max_objs, 2), dtype=np.float32)
      off_lms = np.zeros((self.max_objs, 21 *2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

      if True:
        w, h = left_w / down, left_h / down               
        lms21_down =  lms_left / down    
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        ct_int = (ct_left / down).astype(np.int32)
        for kk in range(21):
          draw_umich_gaussian(hm_lms[kk], (lms21_down[kk]).astype(np.int32), hp_radius)  
          off_lms[0, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int    
        draw_umich_gaussian(hm[0], ct_int, hp_radius)    
        wh[0] = 1. * w, 1. * h
        ind[0] = ct_int[1] * heatmap_w + ct_int[0]
        off_hm[0] = (ct_left / down) - ct_int
        reg_mask[0] = 1
      
      if True:
        w, h = right_w / down, right_h / down   
        lms21_down =  lms_right / down        
        hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        hp_radius = max(0, int(hp_radius))
        ct_int = (ct_right / down).astype(np.int32)
        for kk in range(21):
          draw_umich_gaussian(hm_lms[21+kk], (lms21_down[kk]).astype(np.int32), hp_radius)
          off_lms[1, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int      
        draw_umich_gaussian(hm[1], ct_int, hp_radius)   
        wh[1] = 1. * w, 1. * h
        ind[1] = ct_int[1] * heatmap_w + ct_int[0]
        off_hm[1] = (ct_right / down) - ct_int
        reg_mask[1] = 1

      # address the outscreen aug case.
      if ind[0] >= heatmap_h*heatmap_w or ind[0] <0:
        ind[0] = 0
      if ind[1] >= heatmap_h*heatmap_w or ind[1] <0:
        ind[1] = 0

      ret = {'hm': hm}
      # if self.opt.heatmaps:
      ret.update({'hms': hm_lms})

      ret.update({'valid': reg_mask.astype(np.int64), 'ind': ind})
      ret.update({'wh': wh}) # used for perceptual loss
      if self.opt.off:
        ret.update({'off_lms': off_lms})
        ret.update({'off_hm': off_hm})
      # add the followings for IntagHand format.  
      # ret.update({'rot_point': rot_point.astype(np.float32)})

      ret.update({'input': self.normal(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)})
      ret.update({'image': img.copy()})
      ret.update({'mask': mask.astype(np.float32)})
      ret.update({'mask_left_gt': mask_left_gt.astype(np.float32)})
      ret.update({'mask_right_gt': mask_right_gt.astype(np.float32)})
      ret.update({'dense': dense.astype(np.float32)})
      ret.update({'hms': hms.astype(np.float32)})
      ret.update({'file_id':index})
      if bool_flip: # right is left
        ret.update({'lms_left_gt': hand_dict['right']['joints2d'].astype(np.float32)})
        ret.update({'lms_right_gt': hand_dict['left']['joints2d'].astype(np.float32)})
        ret.update({'verts2d_left_gt': hand_dict['right']['verts2d'].astype(np.float32)})
        ret.update({'verts2d_right_gt': hand_dict['left']['verts2d'].astype(np.float32)})
        ret.update({'joints_left_gt': hand_dict['right']['joints3d'].astype(np.float32)})
        ret.update({'verts_left_gt': hand_dict['right']['verts3d'].astype(np.float32)})
        ret.update({'joints_right_gt': hand_dict['left']['joints3d'].astype(np.float32)})
        ret.update({'verts_right_gt': hand_dict['left']['verts3d'].astype(np.float32)})
      else:
        ret.update({'lms_left_gt': hand_dict['left']['joints2d'].astype(np.float32)})
        ret.update({'lms_right_gt': hand_dict['right']['joints2d'].astype(np.float32)})
        ret.update({'verts2d_left_gt': hand_dict['left']['verts2d'].astype(np.float32)})
        ret.update({'verts2d_right_gt': hand_dict['right']['verts2d'].astype(np.float32)})
        ret.update({'joints_left_gt': hand_dict['left']['joints3d'].astype(np.float32)})
        ret.update({'verts_left_gt': hand_dict['left']['verts3d'].astype(np.float32)})
        ret.update({'joints_right_gt': hand_dict['right']['joints3d'].astype(np.float32)})
        ret.update({'verts_right_gt': hand_dict['right']['verts3d'].astype(np.float32)})
      ret.update({'K_new': hand_dict['left']['camera'].astype(np.float32)})
      # ret.update({'camera_right': hand_dict['right']['camera'].astype(np.float32)})
      # vis
      if False:
        lms_img = draw_lms(img, ret['lms_left_gt'].reshape(1,-1,2))
        cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
        if self.opt.photometric_loss:
          cv2.imwrite('new_mask.png',ret['mask'])              
      return ret
      
    if self.split == 'train' or self.split == 'train_3d' or self.split == 'val' or self.split == 'test':
      self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None, use_pca=False),
                          'left': ManoLayer(mano_path['left'], center_idx=None, use_pca=False)}
      # np.random.seed(317)
      ret, x_img, depth = self.augment_centernet(self.data[index],self.split)
      # dpt, M, com = self.get_pcl(depth*1000)
      img = x_img.copy()
      if self.opt.brightness and np.random.randint(0, 2) == 0:
        # img = data_augment._brightness(x_img, min=0.8, max=1.2)
        img = data_augment.add_noise(x_img.astype(np.float32),
                                     noise=0.0,
                                     scale=255.0,
                                     alpha=0.3, beta=0.05).astype(np.uint8)

      ret.update({'file_id': index})
      ret.update({'dataset': self.data[index]['dataset']})
      if 'id' in self.data[index] and self.split == 'test':
        ret.update({'id': (self.data[index]['id'])}) 
        ret.update({'frame_num': int(self.data[index]['imgpath'][-10:-4])})       
      ret.update({'input': self.normal(img).transpose(2, 0, 1)})
      if depth is not None:
        ret.update({'depth': depth.astype(np.float32).reshape(1,self.opt.size_train[0],self.opt.size_train[1])}) 

      # img_new = (img / 255.)[:, :, ::-1].astype(np.float32)
      ret.update({'image': img.copy()})

      if 'mano_coeff' in self.data[index]:
        ret.update({'joints': (self.data[index]['joints']).astype(np.float32).reshape(1,-1)})
        ret.update({'mano_coeff': self.data[index]['mano_coeff'].astype(np.float32).reshape(1,-1)})
        ret.update({'K': np.array(self.data[index]['K'].astype(np.float32).reshape(1,-1))})

      # lms_img = draw_lms(img, ret['lms_left_gt'].reshape(ret['lms_left_gt'].shape[0],-1,2))
      # cv2.imwrite('local_lms_mixed_img.jpg', lms_img)      
      # cv2.imwrite('new_image.jpg',x_img)
      # if self.opt.photometric_loss:
      #   cv2.imwrite('new_mask.png',img_data['mask'])

      return ret


  def augment_centernet(self, img_data, split):
    assert 'imgpath' in img_data
    img_data_aug = copy.deepcopy(img_data)
    img = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['imgpath']))
    if self.opt.dataset=='RHD': 
      depth = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['depthpath'])) if 'depthpath' in img_data and self.opt.depth else None
      depth = self.depth_two_uint8_to_float(depth[:, :, 2], depth[:, :, 1])  # depth in meters from the camera
    else: # H2O
      depth = cv2.imread(os.path.join(self.opt.pre_fix, img_data_aug['depthpath']), cv2.IMREAD_ANYDEPTH) / 1000. if 'depthpath' in img_data and self.opt.depth else None
    # depth = depth / 1000. # transpose to meters
    # depth = read_depth_img(os.path.join(self.opt.pre_fix, img_data_aug['depthpath'])) if 'depthpath' in img_data and self.opt.depth else None
    # mask[:,:,2] because HO3D has red hand and blue object.
    mask_path = img_data_aug['imgpath'].replace('rgb','mask') if self.opt.dataset == 'H2O' else img_data_aug['imgpath'].replace('color','mask') # for H2O mask file/ RHD
    mask = cv2.imread(os.path.join(self.opt.pre_fix, mask_path))

    # mask = cv2.imread(os.path.join(self.opt.pre_fix, mask_path))[:, :, 2] \
    #   if 'maskpath' in img_data_aug and self.opt.photometric_loss else None # 
    if img is None:
      print('what',os.path.join(self.opt.pre_fix, img_data_aug['imgpath']))    
    img_height, img_width = img.shape[:2]
    if mask is not None:
      mask_height, mask_width = mask.shape[:2]
      if (mask_height != img_height) or (mask_width != img_width):
        mask = cv2.resize(mask,(img_width,img_height))
        
    c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
    s = max(img_height, img_width) * 1.
    rot = 0

    # gts = np.repeat(gts,5,axis=0)
    if 'lms' in img_data_aug:
      lms = np.copy(img_data_aug['lms'][:,:2]).astype(np.float32).reshape(-1,2)
      lms_left = np.copy(img_data_aug['lms'][:21,:2]).astype(np.float32).reshape(-1,2)
      lms_right = np.copy(img_data_aug['lms'][21:,:2]).astype(np.float32).reshape(-1,2)

    if 'joints' in img_data_aug:
      joints_left = np.copy(img_data_aug['joints'][:21,:3]).astype(np.float32).reshape(-1,3)
      joints_right = np.copy(img_data_aug['joints'][21:,:3]).astype(np.float32).reshape(-1,3)

    if 'K' in img_data_aug:
      K = np.copy(img_data_aug['K']).astype(np.float32)
      cx, cy, fx, fy= K[0,2],K[1,2],K[0,0],K[1,1]

    bool_flip = True if np.random.randint(0, 2) == 0 and split == 'train' else False
    # vis
    if False:
        for xy in lms[:]:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 2, (0,255,255), 2)
        cv2.imwrite('img.jpg',img)    

        img_points = projectPoints(
            joints_left, K)
        for xy in img_points[:]:
            cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,0,255), 1)
        cv2.imwrite('img_proj.jpg',img)

    if 'mano_coeff' in img_data_aug:
      gt_trans_left = img_data_aug['mano_coeff'][1:4].reshape(-1,3)
      gt_pose_left = img_data_aug['mano_coeff'][7:52].reshape(-1,45)
      gt_shape_left = img_data_aug['mano_coeff'][52:62].reshape(-1,10)
      gt_orient_left = img_data_aug['mano_coeff'][4:7].reshape(-1,3)
      gt_trans_right = img_data_aug['mano_coeff'][62+1:62+4].reshape(-1,3)
      gt_pose_right = img_data_aug['mano_coeff'][62+7:62+52].reshape(-1,45)
      gt_shape_right = img_data_aug['mano_coeff'][62+52:62+62].reshape(-1,10)
      gt_orient_right = img_data_aug['mano_coeff'][62+4:62+7].reshape(-1,3)            

      hand_dict = {}
      for hand_type in ['left', 'right']:
          params = img_data_aug['mano_coeff'].reshape(1,-1)[:,:62] if hand_type == 'left' else img_data_aug['mano_coeff'].reshape(1,-1)[:,62:]
          handV, handJ = self.mano_layer[hand_type](torch.from_numpy(params[:,4:7]).float(),
                                                  torch.from_numpy(params[:,7:52]).float(),
                                                  torch.from_numpy(params[:,52:62]).float(),
                                                  trans=torch.from_numpy(params[:,1:4]).float(), side = hand_type)
          handV = handV[0].numpy()
          handJ = handJ[0].numpy()
          handV2d = handV @ K.T
          handV2d = handV2d[:, :2] / handV2d[:, 2:]
          handJ2d = handJ @ K.T
          handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]
          if bool_flip:
            handJ2d[:,0] = img.shape[1] - handJ2d[:, 0]
            handV2d[:,0] = img.shape[1] - handV2d[:, 0]
            # here, we want to keep the right absolute position, not aligned flip hand.
            handJ[:, 0] = -handJ[:,0] + handJ[:, 2]/fx * (img_width-2*cx)
            handV[:, 0] = -handV[:,0] + handV[:, 2]/fx * (img_width-2*cx)
          hand_dict[hand_type] = {#'hms': hms,
                                  'verts3d': handV, 'joints3d': handJ,
                                  'verts2d': handV2d, 'joints2d': handJ2d
                                  }
    else: # RHD with no verts
      hand_dict = {}
      for hand_type in ['left', 'right']:
        handJ = joints_left if hand_type == 'left' else joints_right
        handJ2d = lms_left if hand_type == 'left' else lms_right
        handV = None
        handV2d = None
        if bool_flip:
          handJ2d[:,0] = img.shape[1] - handJ2d[:, 0]
          # handV2d[:,0] = img.shape[1] - handV2d[:, 0]
          # here, we want to keep the right absolute position, not aligned flip hand.
          handJ[:, 0] = -handJ[:,0] + handJ[:, 2]/fx * (img_width-2*cx)
          # handV[:, 0] = -handV[:,0] + handV[:, 2]/fx * (img_width-2*cx)
        hand_dict[hand_type] = {#'hms': hms,
                                'verts3d': handV, 'joints3d': handJ,
                                'verts2d': handV2d, 'joints2d': handJ2d
                                }
    # add img noise
    if self.opt.brightness and np.random.randint(0, 2) == 0:
      img = data_augment.add_noise(img.astype(np.float32),
                                    noise=0.0,
                                    scale=255.0,
                                    alpha=0.3, beta=0.05).astype(np.uint8)
    select_hand = img_data_aug['handness'] if 'handness' in img_data_aug else None
    if bool_flip:
      img = cv2.flip(img, 1)
      mask = cv2.flip(mask, 1)
      depth = cv2.flip(depth, 1)
      lms[:,0] = img.shape[1] - lms[:, 0]
      if select_hand is not None:
        select_hand = 1 - select_hand
      # dense = cv2.flip(dense, 1)
    if False:
      # RHD with only one hand
      if bool_flip:
        crop_lms = lms[21:] if select_hand == 0 else lms[:21]
      else:
        crop_lms = lms[:21] if select_hand == 0 else lms[21:]
      twohand_bbox = lms2bbox(crop_lms)
      center = (twohand_bbox[0, 2:] + twohand_bbox[0, :2]) / 2
      center_noise = 5
      c[0] = np.random.randint(low=int(center[0] - center_noise), high=int(center[0] + center_noise))
      c[1] = np.random.randint(low=int(center[1] - center_noise), high=int(center[1] + center_noise))    
      max_size = max(twohand_bbox[0, 2:] - twohand_bbox[0, :2] + 1) 
      min_scale, max_scale = max_size / 0.65 / s, max_size / 0.5 / s
      s = s * np.random.choice(np.arange(min_scale, max_scale, 0.01))   
      rot = np.random.randint(low=-60, high=60) 
    if split == 'train' and True:
      center_noise = 5
      c[0] = np.random.randint(low=int(c[0] - center_noise), high=int(c[0] + center_noise))
      c[1] = np.random.randint(low=int(c[1] - center_noise), high=int(c[1] + center_noise))  
      rot = np.random.randint(low=-60, high=60) 

    trans_input,inv_trans = get_affine_transform(c, s, 0, [self.opt.size_train[0], self.opt.size_train[1]])
    # first, we get cropped img and modified fx,fy,cx,cy in K_img.    
    K_img = K.copy()
    K_img[0][0] = K[0][0]*trans_input[0][0]
    K_img[1][1] = K[1][1]*trans_input[1][1]
    K_img[0][2] = K[0][2]*trans_input[0][0] + trans_input[0][2]
    K_img[1][2] = K[1][2]*trans_input[1][1] + trans_input[1][2]
    cx, cy, fx, fy= K_img[0,2],K_img[1,2],K_img[0,0],K_img[1,1]

    img = cv2.warpAffine(img, trans_input,
                         (self.opt.size_train[0], self.opt.size_train[1]),
                         flags=cv2.INTER_LINEAR)
    if depth is not None:                    
      depth_256 = cv2.warpAffine(depth, trans_input,
                          (int(self.opt.size_train[0]), int(self.opt.size_train[1])),
                          flags=cv2.INTER_NEAREST)     
    if mask is not None:
      mask_256 = cv2.warpAffine(mask, trans_input, (int(self.opt.size_train[0]), int(self.opt.size_train[1])),
                            flags=cv2.INTER_NEAREST)    
    if lms is not None:
      lms = affine_transform_array(lms,trans_input)
      for hand_type in ['left', 'right']:
        hand_dict[hand_type]['joints2d'] = affine_transform_array(hand_dict[hand_type]['joints2d'],trans_input)
        hand_dict[hand_type]['verts2d'] = affine_transform_array(hand_dict[hand_type]['verts2d'],trans_input) if hand_dict[hand_type]['verts2d'] is not None else None

    # then, we rotate cropped img and 3D points along axis-z and keep K_img same.
    if True:
      c = np.array([self.opt.size_train[0]/2, self.opt.size_train[1]/2], dtype=np.float32)
      s = max(self.opt.size_train[0], self.opt.size_train[1]) * 1.
      trans_input1,inv_trans = get_affine_transform(c, s, rot, [self.opt.size_train[0], self.opt.size_train[1]])
      img = cv2.warpAffine(img, trans_input1,
                          (self.opt.size_train[0], self.opt.size_train[1]),
                          flags=cv2.INTER_LINEAR)
      if depth is not None:                    
        depth_256 = cv2.warpAffine(depth_256, trans_input1,
                            (int(self.opt.size_train[0]), int(self.opt.size_train[1])),
                            flags=cv2.INTER_NEAREST)      
      if mask is not None:
        mask_256 = cv2.warpAffine(mask_256, trans_input1, 
                            (int(self.opt.size_train[0]), int(self.opt.size_train[1])),
                              flags=cv2.INTER_NEAREST)   
      if lms is not None:
        lms = affine_transform_array(lms,trans_input1)  
      # rotate 3D joints
      tx, ty = trans_input1[0,2], trans_input1[1,2]
      t0 = (trans_input1[0,0] * cx + trans_input1[0,1] * cy +tx -cx) / (fx+1e-7)
      t1 = (trans_input1[1,0] * cx + trans_input1[1,1] * cy +ty -cy) / (fy+1e-7)
      rot_point = np.array([[np.cos(rot / 180. * np.pi), np.sin(rot / 180. * np.pi), t0],
              [-np.sin(rot / 180. * np.pi), np.cos(rot / 180. * np.pi), t1],
              [0, 0, 1]])
      rot_point[:2,:2] = trans_input1[:2,:2].copy()
      for hand_type in ['left', 'right']:
        hand_dict[hand_type]['joints2d'] = affine_transform_array(hand_dict[hand_type]['joints2d'],trans_input1)
        hand_dict[hand_type]['verts2d'] = affine_transform_array(hand_dict[hand_type]['verts2d'],trans_input1) if hand_dict[hand_type]['verts2d'] is not None else None
        hand_dict[hand_type]['joints3d'] = np.matmul(hand_dict[hand_type]['joints3d'],rot_point.T)
        hand_dict[hand_type]['verts3d'] = np.matmul(hand_dict[hand_type]['verts3d'],rot_point.T) if hand_dict[hand_type]['verts3d'] is not None else None
        ## proj to image for vis
        if False:
          handV2d = hand_dict[hand_type]['joints3d'] @ K_img.T
          handV2d = handV2d[:, :2] / handV2d[:, 2:]
          lms_img = draw_lms(img, handV2d.reshape(1,-1,2))
          cv2.imwrite('local_lms_mixed_img.jpg', lms_img)
          with open('hands_{}.obj'.format(33), 'w') as f:
            for v in hand_dict[hand_type]['verts3d'].reshape(-1,778,3)[0]:
              f.write('v %f %f %f\n'%(v[0],v[1],v[2]))

    if mask_256 is not None and self.opt.dataset == 'H2O':
      # save mask to [64,64]
      ret, mask_256 = cv2.threshold(mask_256, 127, 255, cv2.THRESH_BINARY)
      mask_256 = mask_256.astype(np.float32) / 255
      mask_256 = mask_256[..., 1:]
      # mask_left_gt = (mask[:,:,2]>100).astype(np.uint8) 
      # mask_right_gt = (mask[:,:,1]>100).astype(np.uint8)
      if bool_flip:
        mask_256 = mask_256[..., [1, 0]]   
      mask_64 = cv2.resize(mask_256,(self.opt.size_train[0]//4,self.opt.size_train[0]//4))
      mask_64 = mask_64.transpose(2, 0, 1)
      mask_256 = mask_256.transpose(2, 0, 1)
      mask_right_gt = mask_256[0]
      mask_left_gt = mask_256[1]
    else: # RHD
      mask_left_gt = ((mask_256[:,:,0] > 1) & (mask_256[:,:,0] < 18)).astype(np.uint8) # for RHD
      mask_right_gt = (mask_256[:,:,0] >= 18).astype(np.uint8) # for RHD
      mask_256 = np.stack((mask_right_gt,mask_left_gt),0)
      if bool_flip:
        mask_256 = mask_256[[1, 0],...]   
        mask_right_gt = mask_256[0]
        mask_left_gt = mask_256[1]

    # flip lms to get correct center index.
    if bool_flip:
      lms_left = hand_dict['right']['joints2d']
      lms_right = hand_dict['left']['joints2d']
      # center and valid also flip
      if self.opt.dataset == 'RHD':
        valid_left = 1 if img_data_aug['bboxes'][1] is not None and img_data_aug['lms'][21:,2].sum()>10 else 0  
        valid_right = 1 if img_data_aug['bboxes'][0] is not None and img_data_aug['lms'][:21,2].sum()>10 else 0    
      else:
        valid_left = 1 if img_data_aug['mano_coeff'][62] == 1 else 0   
        valid_right = 1 if img_data_aug['mano_coeff'][0] == 1 else 0
    else:
      lms_left = hand_dict['left']['joints2d']
      lms_right = hand_dict['right']['joints2d']
      if self.opt.dataset == 'RHD':
        valid_left = 1 if img_data_aug['bboxes'][0] is not None and img_data_aug['lms'][:21,2].sum()>10 else 0  
        valid_right = 1 if img_data_aug['bboxes'][1] is not None and img_data_aug['lms'][21:,2].sum()>10 else 0    
      else:
        valid_left = 1 if img_data_aug['mano_coeff'][0] == 1 else 0 
        valid_right = 1 if img_data_aug['mano_coeff'][62] == 1 else 0
    bbox_left = lms2bbox(lms_left)
    ct_left = (bbox_left[0, 2:] + bbox_left[0, :2]) / 2
    left_w, left_h = (bbox_left[:, 2] - bbox_left[:, 0])/0.7, (bbox_left[:, 3] - bbox_left[:, 1])/0.7

    bbox_right = lms2bbox(lms_right)
    ct_right = (bbox_right[0, 2:] + bbox_right[0, :2]) / 2
    right_w, right_h = (bbox_right[:, 2] - bbox_right[:, 0])/0.7, (bbox_right[:, 3] - bbox_right[:, 1])/0.7

    Z_max = 2.5
    Z_min = 0.2
    noise_mask = ((Z_min < depth_256) & (Z_max > depth_256)).astype(np.uint8)
    depth = depth_256 * noise_mask
    masked_hand_depth_left = depth * mask_left_gt
    masked_hand_depth_right = depth * mask_right_gt

    if True: # test rgb only
      # 2.2 convert depth to xyz
      with_normal = True if self.opt.INPUT_FEATURE_NUM == 6 else False
      num_points = self.opt.SAMPLE_NUM
      if valid_left == 1:
        points_xyz_left, normals_left = get_normal(masked_hand_depth_left, K_img, with_normal)
        points_xyz_left = points_xyz_left.reshape(3,-1)
        if with_normal:
          normals_left = normals_left.reshape(-1,3)
        if len(points_xyz_left[2,points_xyz_left[2,:]!=0])!=0:
          mean_dis = points_xyz_left[2,points_xyz_left[2,:]!=0].mean()
          min_dis, max_dis = max(Z_min,mean_dis - 0.08), min(Z_max, mean_dis + 0.08)
          choose_left = ((points_xyz_left[2,:]>min_dis) & (points_xyz_left[2,:]<max_dis))
          choose_left = choose_left.flatten().nonzero()[0]
          tmpl = len(choose_left)
          # if tmpl<80:
          #   print('what')

          if len(choose_left) < 100:
              choose_left = np.zeros((num_points), dtype=np.int64)
              valid_left = 0
          elif len(choose_left) > num_points:
              c_mask = np.zeros(len(choose_left), dtype=int)
              c_mask[:num_points] = 1
              np.random.shuffle(c_mask)
              choose_left = choose_left[c_mask.nonzero()]
          else:
              choose_left = np.pad(choose_left, (0, num_points - len(choose_left)), 'wrap')
        else:
          choose_left = np.zeros((num_points), dtype=np.int64)   
          valid_left = 0
        np.random.shuffle(choose_left)
        hand_points_left = points_xyz_left.transpose(1,0)[choose_left,:]
        if with_normal:
          normals_sampled_left = normals_left[choose_left,:]
          hand_points_left = np.concatenate((hand_points_left,normals_sampled_left),1)
      else:
        choose_left = np.zeros((num_points), dtype=np.int64)
        if with_normal:
          hand_points_left = np.zeros((num_points,6), dtype=np.float32)
        else:
           hand_points_left = np.zeros((num_points,3), dtype=np.float32)

      if valid_right == 1:
        points_xyz_right, normals_right = get_normal(masked_hand_depth_right, K_img, with_normal)
        points_xyz_right = points_xyz_right.reshape(3,-1)
        if with_normal:
          normals_right = normals_right.reshape(-1,3) 
        if len(points_xyz_right[2,points_xyz_right[2,:]!=0])!=0:
          mean_dis = points_xyz_right[2,points_xyz_right[2,:]!=0].mean()
          min_dis, max_dis = max(Z_min,mean_dis - 0.08), min(Z_max, mean_dis + 0.08)
          choose_right = ((points_xyz_right[2,:]>min_dis) & (points_xyz_right[2,:]<max_dis))
          choose_right = choose_right.flatten().nonzero()[0]
          tmpr = len(choose_right)
          # if tmpr<80:
          #   print('what')
          if len(choose_right) < 100:
              choose_right = np.zeros((num_points), dtype=np.int64)
              valid_right = 0
          elif len(choose_right) > num_points:
              c_mask = np.zeros(len(choose_right), dtype=int)
              c_mask[:num_points] = 1
              np.random.shuffle(c_mask)
              choose_right = choose_right[c_mask.nonzero()]
          else:
              choose_right = np.pad(choose_right, (0, num_points - len(choose_right)), 'wrap')
        else:
          choose_right = np.zeros((num_points), dtype=np.int64)   
          valid_right = 0
        np.random.shuffle(choose_right)
        hand_points_right = points_xyz_right.transpose(1,0)[choose_right,:]
        if with_normal:
          normals_sampled_right = normals_right[choose_right,:]
          hand_points_right = np.concatenate((hand_points_right,normals_sampled_right),1)
      else:
        choose_right = np.zeros((num_points), dtype=np.int64)
        if with_normal:
          hand_points_right = np.zeros((num_points,6), dtype=np.float32)
        else:
          hand_points_right = np.zeros((num_points,3), dtype=np.float32)

      if len(choose_left)!= len(choose_right):
        print(len(choose_left),len(choose_right),tmpl,tmpr)

      if False:
        cv2.imwrite('img.jpg',img)
        with open('hand_points_left{}.obj'.format(5), 'w') as f:
          for v in hand_points_left[:]:
            f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
        with open('hand_points_right{}.obj'.format(5), 'w') as f:
          for v in hand_points_right[:]:
            f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
      ##2.7 FPS Sampling
      # if self.opt.sample_strategy=='FPS':
      #   sample_num_level1 = self.opt.sample_num_level1
      #   sample_num_level2 = self.opt.sample_num_level2
      #   if valid_left == 1:
      #   #   #% 1st level
      #     sampled_idx_l1 = np.unique(fps(torch.from_numpy(hand_points_left[:,:3]), None, ratio=0.5, random_start=True))
      #     # sampled_idx_l1 = np.transpose(self.farthest_point_sampling_fast(hand_points_left[:,:3], sample_num_level1))
      #     other_idx = np.setdiff1d(np.arange(num_points), sampled_idx_l1)
      #     new_idx = np.append(sampled_idx_l1, other_idx)
      #     if len(new_idx)!=1024:
      #       print('left',sampled_idx_l1, other_idx,len(sampled_idx_l1),len(other_idx))
      #       print('valid',valid_left,valid_right,bool_flip)
      #       print('shape',len(choose_left),len(choose_right),tmpl)
      #     hand_points_left = hand_points_left[new_idx,:]
      #     choose_left = choose_left[new_idx]
      #     #% 2nd level
      #     sampled_idx_l2 = np.unique(fps(torch.from_numpy(hand_points_left[:sample_num_level1,:3]), None, ratio=0.25, random_start=True))
      #   # sampled_idx_l2 = np.transpose(self.farthest_point_sampling_fast(hand_points_left[:sample_num_level1,:3], sample_num_level2))
      #     other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
      #     new_idx = np.append(sampled_idx_l2, other_idx)
      #     if len(new_idx)!= sample_num_level1:
      #       print('len(lnew_idx): ',len(sampled_idx_l2),len(other_idx))
      #     hand_points_left[:sample_num_level1,:] = hand_points_left[new_idx,:]
      #     choose_left[:sample_num_level1] = choose_left[new_idx]
      #   if valid_right == 1:
      #   #   #% 1st level
      #     sampled_idx_l1 = np.unique(fps(torch.from_numpy(hand_points_left[:,:3]), None, ratio=0.5, random_start=True))
      #   #   # sampled_idx_l1 = np.transpose(self.farthest_point_sampling_fast(hand_points_right[:,:3], sample_num_level1))
      #     other_idx = np.setdiff1d(np.arange(num_points), sampled_idx_l1)
      #     new_idx = np.append(sampled_idx_l1, other_idx)
      #     if len(new_idx)!=1024:
      #       print('right',sampled_idx_l1, other_idx)
      #     hand_points_right = hand_points_right[new_idx,:]
      #     choose_right = choose_right[new_idx]
      #     #% 2nd level
      #     sampled_idx_l2 = np.unique(fps(torch.from_numpy(hand_points_left[:sample_num_level1,:3]), None, ratio=0.25, random_start=True))
      #     # sampled_idx_l2 = np.transpose(self.farthest_point_sampling_fast(hand_points_right[:sample_num_level1,:3], sample_num_level2))
      #     other_idx = np.setdiff1d(np.arange(sample_num_level1), sampled_idx_l2)
      #     new_idx = np.append(sampled_idx_l2, other_idx)
      #     if len(new_idx)!= sample_num_level1:
      #       print('len(rnew_idx): ',len(sampled_idx_l2),len(other_idx))
      #     hand_points_right[:sample_num_level1,:] = hand_points_right[new_idx,:]
      #     choose_right[:sample_num_level1] = choose_right[new_idx]

      if len(choose_left) != len(choose_right):
        print(len(choose_left),len(choose_right))
      choose = np.stack([choose_left,choose_right])
      cloud = np.stack([hand_points_left,hand_points_right])
    else:
        choose = np.zeros((2,1024), dtype=np.int64)
        cloud = np.zeros((2,1024,3), dtype=np.float32)
    # vis
    if False:
      for i in range(len(hand_points_left)):
          jj = cx + hand_points_left[i][0]*fx/hand_points_left[i][2]
          ii = cy + hand_points_left[i][1]*fy/hand_points_left[i][2]
          cv2.circle(depth_256,(int(jj), int(ii)), 1, (0,0,244), 1)
      cv2.imwrite('depth.jpg',depth_256*255)     

    # calculate 1/8 gts
    down = self.opt.down_ratio
    # down = 4 # test for IntagHand format
    heatmap_h, heatmap_w = self.opt.size_train[1] // down, self.opt.size_train[0] // down
    hm = np.zeros((self.num_classes, heatmap_h, heatmap_w), dtype=np.float32)
    hm_lms = np.zeros((42, heatmap_h, heatmap_w), dtype=np.float32)
    wh = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_hm = np.zeros((self.max_objs, 2), dtype=np.float32)
    off_lms = np.zeros((self.max_objs, 21 *2), dtype=np.float32)
    ind = np.zeros((self.max_objs), dtype=np.int64)
    reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

    if valid_left:
      w, h = left_w / down, left_h / down               
      lms21_down =  lms_left / down    
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))
      ct_int = (ct_left / down).astype(np.int32)
      for kk in range(21):
        draw_umich_gaussian(hm_lms[kk], (lms21_down[kk]).astype(np.int32), hp_radius)  
        off_lms[0, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int    
      draw_umich_gaussian(hm[0], ct_int, hp_radius)    
      wh[0] = 1. * w, 1. * h
      ind[0] = ct_int[1] * heatmap_w + ct_int[0]
      off_hm[0] = (ct_left / down) - ct_int
      reg_mask[0] = 1
     
    if valid_right:
      w, h = right_w / down, right_h / down   
      lms21_down =  lms_right / down        
      hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
      hp_radius = max(0, int(hp_radius))
      ct_int = (ct_right / down).astype(np.int32)
      for kk in range(21):
        draw_umich_gaussian(hm_lms[21+kk], (lms21_down[kk]).astype(np.int32), hp_radius)
        off_lms[1, kk * 2: kk * 2 + 2] = lms21_down[kk, :2] - ct_int      
      draw_umich_gaussian(hm[1], ct_int, hp_radius)   
      wh[1] = 1. * w, 1. * h
      ind[1] = ct_int[1] * heatmap_w + ct_int[0]
      off_hm[1] = (ct_right / down) - ct_int
      reg_mask[1] = 1

    # address the outscreen aug case.
    if ind[0] >= heatmap_h*heatmap_w or ind[0] <0:
      ind[0] = 0
    if ind[1] >= heatmap_h*heatmap_w or ind[1] <0:
      ind[1] = 0

    if mask is not None:
      img_data_aug['mask'] = mask_256
      #vis
    if False:
      for xy in lms_left[:]:
          cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,0,255), 1)
      cv2.imwrite('img_new_proj.jpg',img) 
      cv2.rectangle(img, (int(ct_left[0]-left_w/2), int(ct_left[1]-left_h/2)), (int(ct_left[0]+left_w/2), int(ct_left[1]+left_h/2)), (0,255,0), 1)          
      cv2.imwrite('new_left_bbox.png', img)
      cv2.circle(img, (int(ct_left[0]),int(ct_left[1])), 1, (255, 0, 0), 2)
      cv2.imwrite('new_left_bbox.png', img)
      img_points = projectPoints(
          hand_dict['left']['joints3d'], K_img)
      for xy in img_points[:]:
          cv2.circle(img, (int(xy[0]), int(xy[1])), 1, (0,255,0), 1)
      cv2.imwrite('img_proj.jpg',img)
    
                      
    ret = {'hm': hm}
    # if self.opt.heatmaps:
    ret.update({'hms': hm_lms})

    ret.update({'valid': reg_mask.astype(np.float32), 'ind': ind})
    ret.update({'lms': lms.astype(np.float32)})
    ret.update({'wh': wh}) # used for perceptual loss
    ret.update({'K_new': K_img.astype(np.float32)})
    ret.update({'mask': mask_256.astype(np.float32)})
    ret.update({'mask_left_gt': mask_left_gt.astype(np.float32)})
    ret.update({'mask_right_gt': mask_right_gt.astype(np.float32)})
    if self.opt.off:
      ret.update({'off_lms': off_lms})
      ret.update({'off_hm': off_hm})
    # add the followings for IntagHand format.  
    if bool_flip: # right is left
      ret.update({'lms_left_gt': hand_dict['right']['joints2d'].astype(np.float32)})
      ret.update({'lms_right_gt': hand_dict['left']['joints2d'].astype(np.float32)})
      ret.update({'joints_left_gt': hand_dict['right']['joints3d'].astype(np.float32)})
      ret.update({'joints_right_gt': hand_dict['left']['joints3d'].astype(np.float32)})
      if self.opt.dataset == 'H2O':
        ret.update({'verts2d_left_gt': hand_dict['right']['verts2d'].astype(np.float32)})
        ret.update({'verts2d_right_gt': hand_dict['left']['verts2d'].astype(np.float32)})
        ret.update({'verts_left_gt': hand_dict['right']['verts3d'].astype(np.float32)})
        ret.update({'verts_right_gt': hand_dict['left']['verts3d'].astype(np.float32)})
    else:
      ret.update({'lms_left_gt': hand_dict['left']['joints2d'].astype(np.float32)})
      ret.update({'lms_right_gt': hand_dict['right']['joints2d'].astype(np.float32)})
      ret.update({'joints_left_gt': hand_dict['left']['joints3d'].astype(np.float32)})
      ret.update({'joints_right_gt': hand_dict['right']['joints3d'].astype(np.float32)})
      if self.opt.dataset == 'H2O':
        ret.update({'verts2d_left_gt': hand_dict['left']['verts2d'].astype(np.float32)})
        ret.update({'verts2d_right_gt': hand_dict['right']['verts2d'].astype(np.float32)})
        ret.update({'verts_left_gt': hand_dict['left']['verts3d'].astype(np.float32)})
        ret.update({'verts_right_gt': hand_dict['right']['verts3d'].astype(np.float32)})
    ret.update({'rot_point': rot_point.astype(np.float32)})
    ret.update({'choose': choose.astype(np.int64)})
    ret.update({'cloud': cloud.astype(np.float32)})
    if select_hand is not None:
      ret.update({'select': np.array(select_hand)})
    return ret, img, depth_256

