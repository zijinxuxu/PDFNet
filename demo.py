from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import _init_paths

import os
import sys
import math

import torch
import torch.utils.data
from opts import opts
from models.model import create_model
from models.networks.intaghand_model import load_model_intag
from logger import Logger
# from datasets.artificial import ArtificialDataset
from datasets.interhand import InterHandDataset
from datasets.joint_dataset import JointDataset
from trains.simplified import SimplifiedTrainer
from torch.utils.data.sampler import *
from lib.utils.utils import load_model, save_model
import time
import torch.nn.functional as F

import random
import numpy as np
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
import cv2
from lib.models.networks.manolayer import ManoLayer, rodrigues_batch
from lib.models.networks.mano_utils import mano_two_hands_renderer
from lib.utils.image import get_affine_transform, affine_transform, affine_transform_array
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.models.hand3d.Mano_render import ManoRender

def get_dataset(task):
  class Dataset(JointDataset, InterHandDataset):
    pass         
  return Dataset

# import torch.distributed as dist
def seed_torch(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED']=str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.deterministic = True

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

def main(opt):
  # setup 
  seed_torch(opt.seed)
  # torch.manual_seed(opt.seed)
  # torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark
  Dataset = get_dataset(opt.task)
  opt = opts.update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

  render = ManoRender(opt).cuda().eval()
  renderer = mano_two_hands_renderer(img_size=(384,384), device='cuda')
  mano_path = {'left': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_LEFT.pkl'),
              'right': os.path.join('/home/zijinxuxu/codes/SMHR-InterHand/lib/models/hand3d/mano_core', 'MANO_RIGHT.pkl')}            
  mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None, use_pca=False),
                      'left': ManoLayer(mano_path['left'], center_idx=None, use_pca=False)}
  print('Creating model...')
  model = load_model_intag(opt)
  if opt.load_model != '':
      model = load_model(model, opt.load_model)
  model.cuda().eval()
  # base_dir = '/media/zijinxuxu/Seagate Backup Plus Drive/Hands_data/OneHand10K/Evaluation_data'
  base_dir = 'assets/H2O/color'
  img_list = []
  fileid_list = os.listdir(base_dir)
  for fileid in fileid_list:
      post_fix = fileid.split('.')[1]
      if post_fix != 'jpg' and post_fix != 'png':
        continue
      # fileid = fileid.split('.')[0]          
      img_rgb_path = os.path.join(base_dir, fileid) # v3 is .jpg  
      img_list.append(img_rgb_path)
  # img_list = glob.glob('/home/zijinxuxu/Downloads/egohands/egohands_data/_LABELLED_SAMPLES/CARDS_COURTYARD_S_H/*.jpg')
  # img_list = sorted(glob.glob('/mnt/SSD/AFLW/AFLW2000/*.jpg'))
  # img_list = sorted(glob.glob('/mnt/SSD/LS3D/LS3D-W/300W-Testset-3D/*.png'))
  mean = np.array([0.485, 0.456, 0.406],
                  dtype=np.float32).reshape(1, 1, 3)
  std = np.array([0.229, 0.224, 0.225],
                 dtype=np.float32).reshape(1, 1, 3)

  out = 'outputs'
  if not os.path.exists(out):
    os.makedirs(out)

  with torch.no_grad():
    # intrins = np.load('/home/zijinxuxu/codes/SMHR-InterHand/assets/capture_data/projs.npy')
    for i, img_file in enumerate(img_list):
      # print(i)
      image = cv2.imread(img_file)
      depth_file = img_file.replace('color','depth')
      # depth = read_depth_img(depth_file)
      depth = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH) / 1000.
      # image = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
      # depth = cv2.rotate(depth,cv2.ROTATE_90_CLOCKWISE)
      hand_pose_mano_path = depth_file.replace('png','txt')
      if False:
        hand_pose_mano = np.loadtxt(hand_pose_mano_path) # (124,) 
      else:
         hand_pose_mano = None
      cam_fx, cam_fy, cam_cx, cam_cy = 636.6593017578125, 636.251953125, 635.283881879317, 366.8740353496978
      K = np.array([[cam_fx,0,cam_cx],[0,cam_fy,cam_cy],[0,0,1]])
      # K = intrins[i]
      cx, cy = K[0,2], K[1,2]
      K[0,2] = cy
      K[1,2] = cx
      img_height, img_width = image.shape[:2]
      c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
      s = max(img_height, img_width) * 1.
      rot = 0
      trans_input,inv_trans = get_affine_transform(c, s, rot, [opt.input_res, opt.input_res])
      # first, we get cropped img and modified fx,fy,cx,cy in K_img.    
      K_img = K.copy()
      K_img[0][0] = K[0][0]*trans_input[0][0]
      K_img[1][1] = K[1][1]*trans_input[1][1]
      K_img[0][2] = K[0][2]*trans_input[0][0] + trans_input[0][2]
      K_img[1][2] = K[1][2]*trans_input[1][1] + trans_input[1][2]
      cx, cy, fx, fy= K_img[0,2],K_img[1,2],K_img[0,0],K_img[1,1]
      if hand_pose_mano is not None:
        hand_dict = {}
        for hand_type in ['left', 'right']:
            if opt.dataset == 'H2O':
              params = hand_pose_mano.reshape(1,-1)[:,:62] if hand_type == 'left' else hand_pose_mano.reshape(1,-1)[:,62:]
              handV, handJ = mano_layer[hand_type](torch.from_numpy(params[:,4:7]).float(),
                                                      torch.from_numpy(params[:,7:52]).float(),
                                                      torch.from_numpy(params[:,52:62]).float(),
                                                      trans=torch.from_numpy(params[:,1:4]).float(), side = hand_type)
              handV = handV[0].numpy()
              handJ = handJ[0].numpy()                                    
            else: # H2O3D
              coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
              params = hand_pose_mano.reshape(1,-1)[:,:61] if hand_type == 'left' else hand_pose_mano.reshape(1,-1)[:,61:]
              handV, handJ = mano_layer[hand_type](torch.from_numpy(params[:,:3]).float(),
                                                      torch.from_numpy(params[:,3:48]).float(),
                                                      torch.from_numpy(params[:,48:58]).float(),
                                                      trans=torch.from_numpy(params[:,58:61]).float(), side = hand_type)
              handV = handV[0].numpy().dot(coord_change_mat.T)
              handJ = handJ[0].numpy().dot(coord_change_mat.T)                                       

            handV2d = handV @ K_img.T
            handV2d = handV2d[:, :2] / handV2d[:, 2:]
            handJ2d = handJ @ K_img.T
            handJ2d = handJ2d[:, :2] / handJ2d[:, 2:]
            hand_dict[hand_type] = {#'hms': hms,
                                    'verts3d': handV, 'joints3d': handJ,
                                    'verts2d': handV2d, 'joints2d': handJ2d
                                    }
            
      image = cv2.warpAffine(image, trans_input,
                          (int(opt.input_res), int(opt.input_res)),
                          flags=cv2.INTER_LINEAR)
      valid = np.array([[1,1]])
      save_img_0 = image.copy()
      pre_img = preprocess(image, mean, std)
      pre_img = torch.from_numpy(pre_img).permute(2, 0, 1).unsqueeze(0).cuda()
      if False: # just for iphone camera
        img_height, img_width = depth.shape[:2]
        c = np.array([img_width / 2., img_height / 2.], dtype=np.float32)
        s = max(img_height, img_width) * 1.
        rot = 0
        trans_input,inv_trans = get_affine_transform(c, s, rot, [opt.input_res, opt.input_res])
      if depth is not None:                    
        depth_256 = cv2.warpAffine(depth, trans_input,
                            (int(opt.input_res), int(opt.input_res)),
                            flags=cv2.INTER_NEAREST)    
      folder, fname = img_file.split('/')[-2:]
      folder = os.path.join(out, folder)
      if not os.path.exists(folder):
        os.makedirs(folder)

      result, paramsDict, handDictList, otherInfo = model(pre_img, None, None, torch.from_numpy(depth_256).cuda(), None, torch.from_numpy(K_img).cuda(), valid)
      center_hm = _sigmoid(otherInfo['ret']['hm']) 

      if True:
        chms = center_hm.clone().detach()
        score = 0.5
        chms = _nms(chms, 5)
        K = 1
        topk_scores, pred_ind_left, topk_ys, topk_xs = _topk(chms[:,:1,:,:], K)  
        topk_scores, pred_ind_right, topk_ys, topk_xs = _topk(chms[:,1:,:,:], K)      
      ind_left = pred_ind_left 
      ind_right = pred_ind_right 

      verts_left_pred_off = result['verts3d']['left']
      verts_right_pred_off = result['verts3d']['right']
      joints_left_pred_off = torch.matmul(render.MANO_L.full_regressor, verts_left_pred_off) 
      joints_right_pred_off = torch.matmul(render.MANO_R.full_regressor, verts_right_pred_off)  
      if True:
        root_z_left = 0.4 + paramsDict['root']['left'][:,0] / 100
        root_z_right = 0.4 + paramsDict['root']['right'][:,0] / 100
        root_xy_left = paramsDict['root']['left'][:,1:] / 100
        root_xy_right = paramsDict['root']['right'][:,1:] / 100  
        # root_left_pred = torch.stack((root_xy_left[:,0], root_xy_left[:,1], root_z_left),1).unsqueeze(1)
        # root_right_pred = torch.stack((root_xy_right[:,0], root_xy_right[:,1], root_z_right),1).unsqueeze(1)
        root_left_pred = render.get_uv_root_3d(ind_left, root_xy_left, root_z_left,torch.from_numpy(K_img).cuda().unsqueeze(0))
        root_right_pred = render.get_uv_root_3d(ind_right, root_xy_right, root_z_right, torch.from_numpy(K_img).cuda().unsqueeze(0))
        joints_left_pred = joints_left_pred_off + root_left_pred 
        joints_right_pred = joints_right_pred_off + root_right_pred 
        lms_left_pred_proj = render.get_Landmarks_new(joints_left_pred,torch.from_numpy(K_img).cuda().unsqueeze(0))
        lms_right_pred_proj = render.get_Landmarks_new(joints_right_pred,torch.from_numpy(K_img).cuda().unsqueeze(0))
        verts_left_pred = verts_left_pred_off + root_left_pred #if mode == 'val' or mode == 'test' else verts_left_pred_off + root_left_gt
        verts_right_pred = verts_right_pred_off + root_right_pred #if mode == 'val' or mode == 'test' else verts_right_pred_off + root_right_gt
        verts2d_left_pred_proj = render.get_Landmarks_new(verts_left_pred,torch.from_numpy(K_img).cuda().unsqueeze(0))
        verts2d_right_pred_proj = render.get_Landmarks_new(verts_right_pred,torch.from_numpy(K_img).cuda().unsqueeze(0))   


      # vis
      if True:
        # file_id = img_file[-10:-4]
        file_id = img_file.split('/')[-1][:-4] # iphone camera
        cv2.imwrite('%s/mask_lr_%s.jpg' % (folder,file_id), (otherInfo['mask'][0,1].detach().cpu().numpy()*255 + otherInfo['mask'][0,0].detach().cpu().numpy()*255)[84:-84,:])
        # lms_vis_left = lms_left_pred_proj[0]
        # for id in range(len(lms_vis_left)):
        #   cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (0,0,255), 2)
        # lms_vis_right = lms_right_pred_proj[0]
        # for id in range(len(lms_vis_right)):
        #   cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (0,255,0), 2)
        # if hand_pose_mano is not None:
        #   lms_vis_left = hand_dict['left']['joints2d']
        #   for id in range(len(lms_vis_left)):
        #     cv2.circle(save_img_0, (int(lms_vis_left[id,0]), int(lms_vis_left[id,1])), 2, (255,0,0), 2)
        #   lms_vis_right = hand_dict['right']['joints2d']
        #   for id in range(len(lms_vis_right)):
        #     cv2.circle(save_img_0, (int(lms_vis_right[id,0]), int(lms_vis_right[id,1])), 2, (255,0,0), 2)  
        # cv2.imwrite('%s/lms_lr_%s.jpg' %(folder,file_id), save_img_0[84:-84,:,:])
        kps_left_img = showHandJoints(save_img_0,lms_left_pred_proj[0].detach().cpu().numpy())
        showHandJoints(kps_left_img,lms_right_pred_proj[0].detach().cpu().numpy(),'%s/bones_lr_%s.jpg' % (folder,file_id))
        # render two hand mano
        # load from obj
        if False:
          lmesh = np.zeros((778,3))
          rmesh = np.zeros((778,3))
          with open('/home/zijinxuxu/codes/SMHR-InterHand/assets/H2O/depth/figure/gt_hands_l0.obj', 'r') as f:
            line = f.readlines()
            for idx in range(778):
              lmesh[idx,0] = float(np.array(line)[idx].split()[1])
              lmesh[idx,1] = float(np.array(line)[idx].split()[2])
              lmesh[idx,2] = float(np.array(line)[idx].split()[3])
          with open('/home/zijinxuxu/codes/SMHR-InterHand/assets/H2O/depth/figure/gt_hands_r0.obj', 'r') as f:
            line = f.readlines()
            for idx in range(778):
              rmesh[idx,0] = float(np.array(line)[idx].split()[1])
              rmesh[idx,1] = float(np.array(line)[idx].split()[2])
              rmesh[idx,2] = float(np.array(line)[idx].split()[3])
          img_out, mask_out = renderer.render_rgb(cameras=torch.from_numpy(K_img).float().cuda().unsqueeze(0),
                                                  v3d_left=torch.from_numpy(lmesh).cuda().unsqueeze(0).float(),
                                                  v3d_right=torch.from_numpy(rmesh).cuda().unsqueeze(0).float())
        # img_out, mask_out = renderer.render_rgb(cameras=torch.from_numpy(K_img).float().cuda().unsqueeze(0),
        #                                         v3d_left=verts_left_pred.float(),
        #                                         v3d_right=verts_right_pred.float())
        if True: # render two hand mesh 
          img_out, mask_out = renderer.render_rgb(cameras=torch.from_numpy(K_img).float().cuda().unsqueeze(0),
                                                  v3d_left=verts_left_pred.float(),
                                                  v3d_right=verts_right_pred.float())
        img_out = img_out[0].detach().cpu().numpy() * 255
        mask_out = mask_out[0].detach().cpu().numpy()[..., np.newaxis]

        img_out = img_out * mask_out + image * (1 - mask_out)
        img_out = img_out.astype(np.uint8)
        cv2.imwrite('%s/render_%s.jpg' %(folder,file_id), img_out[84:-84,:,:])
        if False:
          # # for rendering .obj
          Faces_l = render.MANO_L.faces.astype(np.int32)
          Faces_r = render.MANO_R.faces.astype(np.int32)
          vis_verts_left = verts_left_pred.reshape(-1,778,3).detach().cpu().numpy()
          vis_verts_right = verts_right_pred.reshape(-1,778,3).detach().cpu().numpy()

          k = 0 # which one in batch.
          if valid[0][0]==1: # left
            with open('%s/models_l_%s.obj' % (folder,file_id), 'w') as f:
              for v in vis_verts_left[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_l+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 
            if hand_pose_mano is not None:
              with open('%s/gt_models_l_%s.obj' % (folder,file_id), 'w') as f:
                for v in hand_dict['left']['verts3d']:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_l+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2]))   
          if valid[0][1]==1: # right
            with open('%s/models_r_%s.obj' % (folder,file_id), 'w') as f:
              for v in vis_verts_right[k]:
                f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
              for face in Faces_r+1:
                f.write('f %f %f %f\n'%(face[0],face[1],face[2]))     
            if hand_pose_mano is not None:
              with open('%s/gt_models_r_%s.obj' % (folder,file_id), 'w') as f:
                for v in hand_dict['right']['verts3d']:
                  f.write('v %f %f %f\n'%(v[0],v[1],v[2]))
                for face in Faces_r+1:
                  f.write('f %f %f %f\n'%(face[0],face[1],face[2])) 

def preprocess(image, mean, std):
  return (image.astype(np.float32) / 255. - mean) / std

def _nms(heat, kernel=3):
  pad = (kernel - 1) // 2
  if kernel == 2:
    hm_pad = F.pad(heat, [0, 1, 0, 1])
    hmax = F.max_pool2d(hm_pad, (kernel, kernel), stride=1, padding=pad)
  else:
    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
  keep = (hmax == heat).float()
  return heat * keep

def _topk(scores, K):
    b, c, h, w = scores.size()
    assert c == 1
    topk_scores, topk_inds = torch.topk(scores.view(b, -1), K)

    topk_inds = topk_inds % (h * w)
    topk_ys = (topk_inds // w).int().float()
    topk_xs = (topk_inds % w).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs


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

    gtIn = np.round(gtIn).astype(np.int32)

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

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
