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

  def read_depth_img(self, depth_filename):
      """Read the depth image in dataset and decode it"""
      # depth_filename = os.path.join(base_dir, split, seq_name, 'depth', file_id + '.png')

      # _assert_exist(depth_filename)

      depth_scale = 0.00012498664727900177
      depth_img = cv2.imread(depth_filename)

      dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
      dpt = dpt * depth_scale

      return dpt
      
  def __getitem__(self, index):
    mano_path = {'left': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_LEFT.pkl'),
                'right': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_RIGHT.pkl')}            
    if self.opt.dataset == 'InterHandNew':
      pass


  def augment_centernet(self, img_data, split):
      pass

