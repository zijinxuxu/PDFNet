import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.nn.init as init
# from dataset.dataset_utils import IMG_SIZE
# from utils.utils import projection_batch
# from models.networks.manolayer import ManoLayer
# from models.networks.model_zoo import get_hrnet, conv1x1, conv3x3, deconv3x3, weights_init, GCN_vert_convert, build_fc_layer, Bottleneck

# from utils.config import load_cfg

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
import torch
import torch.nn as nn
import math
from lib.utils.utils import group_points_2, group_points
import cv2
from lib.utils.utils import get_normal

nstates_plus_1 = [64,64,128]
nstates_plus_2 = [128,128,256]
# nstates_plus_3 = [256,512,1024,1024,512]
nstates_plus_3 = [512,512,1024,1024,512]

class PointNet_Plus(nn.Module):
    def __init__(self, opt):
        super(PointNet_Plus, self).__init__()
        self.num_outputs = opt.PCA_SZ
        self.knn_K = opt.knn_K
        self.ball_radius2 = opt.ball_radius2
        self.sample_num_level1 = opt.sample_num_level1
        self.sample_num_level2 = opt.sample_num_level2
        self.INPUT_FEATURE_NUM = opt.INPUT_FEATURE_NUM
        self.opt = opt
        channel_img1 = [3,64,256]#512
        self.sft0 = SFTLayer(3,3)
        self.sft1 = SFTLayer(131,64)
        self.sft2 = SFTLayer(259,256)
        channel_img = [0,0,0]#512 no img test

        self.netR_1 = nn.Sequential(
            # B*(INPUT_FEATURE_NUM+channel_img)*sample_num_level1*knn_K
            nn.Conv2d(self.INPUT_FEATURE_NUM+channel_img[0], nstates_plus_1[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[0]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[0], nstates_plus_1[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[1]),
            nn.ReLU(inplace=True),
            # B*64*sample_num_level1*knn_K
            nn.Conv2d(nstates_plus_1[1], nstates_plus_1[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_1[2]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level1*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # nn.AvgPool2d((1,self.knn_K),stride=1)
            # B*128*sample_num_level1*1
        )
        
        self.netR_2 = nn.Sequential(
            # B*(131+channel_img)*sample_num_level2*knn_K
            nn.Conv2d(3+nstates_plus_1[2]+channel_img[1], nstates_plus_2[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[0]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[0], nstates_plus_2[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[1]),
            nn.ReLU(inplace=True),
            # B*128*sample_num_level2*knn_K
            nn.Conv2d(nstates_plus_2[1], nstates_plus_2[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_2[2]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*knn_K
            nn.MaxPool2d((1,self.knn_K),stride=1)
            # nn.AvgPool2d((1,self.knn_K),stride=1)
            # B*256*sample_num_level2*1
        )
        
        self.netR_3 = nn.Sequential(
            # B*(259+channel_img)*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2]+channel_img[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # nn.AvgPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )
        
        self.netR_FC = nn.Sequential(
            # B*1024
            nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )
    def forward(self, points, emb, choose): # points[b,1024,3],emb[b,3,256,256]/[b,64,128,128]/[b,256,64,64],choose[b,1024]
        # cat layer0: 256x256
        point_wise_emb_l0 = _tranpose_and_gather_feat(emb[0], choose) #[b,1024,3]
        # points = torch.cat((points, point_wise_emb_l0),2) # [b,1024,6+3]
        points = self.sft0((points.transpose(1,2), point_wise_emb_l0))
        x, y = group_points(points, self.opt) 
        # get pyramid index[256,128,64] / [512,256,128]
        choose_1_2 = (choose // self.opt.default_resolution //2)*(self.opt.default_resolution//2)+choose % self.opt.default_resolution //2
        choose_1_4 = (choose // self.opt.default_resolution //4)*(self.opt.default_resolution//4)+choose % self.opt.default_resolution //4
        point_wise_emb_l1 = _tranpose_and_gather_feat(emb[1], choose_1_2[:,:self.sample_num_level1]) # [b,512,64]
        point_wise_emb_l2 = _tranpose_and_gather_feat(emb[2], choose_1_4[:,:self.sample_num_level2]) # [b,128,256]

        ### during fusion, we can stand img_emb alone, just gather or feed into netR_1 together.
        # x: B*(INPUT_FEATURE_NUM+3)*sample_num_level1*knn_K, y: B*3*sample_num_level1*1, z: B*img_dim*sample_num_level1*1
        x = self.netR_1(x)
        # B*128*sample_num_level1*1
        x = torch.cat((y, x),1).squeeze(-1) # here, we can cat emb after x,y.
        # B*(3+128)*sample_num_level1
        # x = torch.cat((x, point_wise_emb_l1.transpose(1,2)),1) # add img_f here
        x = self.sft1((x, point_wise_emb_l1)).transpose(1,2) # add img_f here
        # B*(3+128+64)*sample_num_level1
        inputs_level2, inputs_level2_center = group_points_2(x, self.sample_num_level1, self.sample_num_level2, self.knn_K, self.ball_radius2)
        # B*(131+64)*sample_num_level2*knn_K, B*3*sample_num_level2*1, B*img_dim*sample_num_level2*1
        
        # B*131*sample_num_level2*knn_K
        x = self.netR_2(inputs_level2)
        # B*256*sample_num_level2*1
        x = torch.cat((inputs_level2_center, x),1) # not add img_f here
        # x = torch.cat((inputs_level2_center, x, point_wise_emb_l2.transpose(1,2).unsqueeze(3)),1) # add img_f here
        x = self.sft2((x.squeeze(-1), point_wise_emb_l2)).transpose(1,2).unsqueeze(3) # add img_f here
        ## use PFT block to aggregate color and depth features
        # x = a * x + b
        # B*(259+256)*sample_num_level2*1
        
        x = self.netR_3(x)
        # B*1024*1*1
        x = x.view(-1,1,nstates_plus_3[2])
        # B*1024
        # x = self.netR_FC(x)
        # B*num_outputs
        
        return x
    
class noop(nn.Module):
    def forward(self, x):
        return x

def build_activate_layer(actType):
    if actType == 'relu':
        return nn.ReLU(inplace=True)
    elif actType == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif actType == 'elu':
        return nn.ELU(inplace=True)
    elif actType == 'sigmoid':
        return nn.Sigmoid()
    elif actType == 'tanh':
        return nn.Tanh()
    elif actType == 'noop':
        return noop()
    else:
        raise RuntimeError('no such activate layer!')


def weights_init(layer):
    classname = layer.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(layer.weight.data)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, 0.0)

def conv1x1(in_channels, out_channels, stride=1, bn_init_zero=False, actFun='relu'):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.constant_(bn.weight, 0. if bn_init_zero else 1.)
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
              build_activate_layer(actFun),
              bn]
    return nn.Sequential(*layers)

#############################
# SFTGAN (pytorch version)
#############################


class SFTLayer(nn.Module):
    def __init__(self, c_fea, c_cond):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(c_cond, c_cond, 1)
        self.SFT_scale_conv1 = nn.Conv2d(c_cond, c_fea, 1)
        self.SFT_shift_conv0 = nn.Conv2d(c_cond, c_cond, 1)
        self.SFT_shift_conv1 = nn.Conv2d(c_cond, c_fea, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = x[0].unsqueeze(3)
        cond = x[1].transpose(1,2).unsqueeze(3)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return (fea * (scale + 1) + shift).transpose(1,2).squeeze(-1)


class ResBlock_SFT(nn.Module):
    def __init__(self):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer()
        self.conv0 = nn.Conv2d(1024, 1024, 1)
        self.sft1 = SFTLayer()
        self.conv1 = nn.Conv2d(1024, 1024, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1((fea, x[1]))
        fea = self.conv1(fea)
        return (x[0] + fea, x[1])  # return a tuple containing features and conditions


class SFT_Net(nn.Module):
    def __init__(self):
        super(SFT_Net, self).__init__()
        # self.conv0 = nn.Conv2d(3, 64, 3, 1, 1)

        sft_branch = []
        for i in range(16):
            sft_branch.append(ResBlock_SFT())
        sft_branch.append(SFTLayer())
        sft_branch.append(nn.Conv2d(1024, 1024, 1))
        self.sft_branch = nn.Sequential(*sft_branch)

        # self.HR_branch = nn.Sequential(
        #     nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
        #     nn.Conv2d(64, 256, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
        #     nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(True), nn.Conv2d(64, 3, 3, 1, 1))

        self.CondNet = nn.Sequential(
            nn.Conv2d(1024, 2048, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(2048, 2048, 1),
            nn.LeakyReLU(0.1, True), nn.Conv2d(2048, 2048, 1), nn.LeakyReLU(0.1, True),
            nn.Conv2d(2048, 2048, 1), nn.LeakyReLU(0.1, True), nn.Conv2d(2048, 512, 1))

    def forward(self, x):
        # x[0]: img; x[1]: seg
        cond = self.CondNet(x[1])
        # fea = self.conv0(x[0])
        res = self.sft_branch((x[0], cond))
        fea = x[0] + res
        # out = self.HR_branch(fea)
        return fea
    
class ResNetSimple_decoder(nn.Module):
    def __init__(self, expansion=4,
                 fDim=[256, 256, 256, 256], direction=['flat', 'up', 'up', 'up'],
                 out_dim=3,up_scale=False):
        super(ResNetSimple_decoder, self).__init__()
        self.models = nn.ModuleList()
        fDim = [512 * expansion] + fDim
        for i in range(len(direction)):
            kernel_size = 1 if direction[i] == 'flat' else 3
            self.models.append(self.make_layer(fDim[i], fDim[i + 1], direction[i], kernel_size=kernel_size))

        if up_scale == False: ## used for hms
            self.final_layer = nn.Sequential(
                nn.Conv2d(in_channels=fDim[-1],out_channels=out_dim,kernel_size=1,stride=1,padding=0),
            )      
        else: ## used for mask
            self.final_layer = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=fDim[-1],out_channels=out_dim,kernel_size=1,stride=1,padding=0),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            )

    def make_layer(self, in_dim, out_dim,
                   direction, kernel_size=3, relu=True, bn=True):
        assert direction in ['flat', 'up']
        assert kernel_size in [1, 3]
        if kernel_size == 3:
            padding = 1
        elif kernel_size == 1:
            padding = 0

        layers = []
        if direction == 'up':
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=1, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.BatchNorm2d(out_dim))

        return nn.Sequential(*layers)

    def forward(self, x):
        fmaps = []
        for i in range(len(self.models)):
            x = self.models[i](x)
            fmaps.append(x)
        x = self.final_layer(x)
        return x, fmaps

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

def depth2pcl(depth_256,mask,K_img,valid):
    depth_256 = depth_256.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8)
    K_img = K_img.detach().cpu().numpy().astype(np.float32)
    if mask is not None:
      # save mask to [64,64]
      mask_right_gt = cv2.resize(mask[0,0],(depth_256.shape[-1],depth_256.shape[-2]))
      mask_left_gt = cv2.resize(mask[0,1],(depth_256.shape[-1],depth_256.shape[-2]))
    #   ret, mask_256 = cv2.threshold(mask_256, 127, 255, cv2.THRESH_BINARY)
    #   mask_256 = mask_256.astype(np.float32) / 255
    #   mask_256 = mask_256[..., 1:]
      # mask_left_gt = (mask[:,:,2]>100).astype(np.uint8) 
      # mask_right_gt = (mask[:,:,1]>100).astype(np.uint8)
    #   if bool_flip:
    #     mask_256 = mask_256[..., [1, 0]]   
    #   mask_64 = cv2.resize(mask_256,(self.opt.size_train[0]//4,self.opt.size_train[0]//4))
    #   mask_64 = mask_64.transpose(2, 0, 1)
    #   mask_256 = mask_256.transpose(2, 0, 1)
    #   mask_right_gt = mask_256[0]
    #   mask_left_gt = mask_256[1]
    Z_max = 2.5
    Z_min = 0.2
    noise_mask = ((Z_min < depth_256) & (Z_max > depth_256)).astype(np.uint8)
    depth = depth_256 * noise_mask
    masked_hand_depth_left = depth * mask_left_gt
    masked_hand_depth_right = depth * mask_right_gt

    if True: # test rgb only
      # 2.2 convert depth to xyz
      with_normal = False# if self.opt.INPUT_FEATURE_NUM == 6 else False
      num_points = 1024 #self.opt.SAMPLE_NUM
      if valid[0,0] == 1:
        points_xyz_left, normals_left = get_normal(masked_hand_depth_left.squeeze(), K_img, with_normal)
        points_xyz_left = points_xyz_left.reshape(3,-1)
        if with_normal:
          normals_left = normals_left.reshape(-1,3)
        if len(points_xyz_left[2,points_xyz_left[2,:]!=0])!=0:
          mean_dis = points_xyz_left[2,points_xyz_left[2,:]!=0].mean()
          min_dis, max_dis = max(Z_min,mean_dis - 0.08), min(Z_max, mean_dis + 0.08)
          choose_left = ((points_xyz_left[2,:]>min_dis) & (points_xyz_left[2,:]<max_dis))
          choose_left = choose_left.flatten().nonzero()[0]
          tmpl = len(choose_left)
          if tmpl<80:
            print('what')

          if len(choose_left) < 10:
              choose_left = np.zeros((num_points), dtype=np.int64)
          elif len(choose_left) > num_points:
              c_mask = np.zeros(len(choose_left), dtype=int)
              c_mask[:num_points] = 1
              np.random.shuffle(c_mask)
              choose_left = choose_left[c_mask.nonzero()]
          else:
              choose_left = np.pad(choose_left, (0, num_points - len(choose_left)), 'wrap')
        else:
          choose_left = np.zeros((num_points), dtype=np.int64)   
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

      if valid[0,1] == 1:
        points_xyz_right, normals_right = get_normal(masked_hand_depth_right.squeeze(), K_img, with_normal)
        points_xyz_right = points_xyz_right.reshape(3,-1)
        if with_normal:
          normals_right = normals_right.reshape(-1,3) 
        if len(points_xyz_right[2,points_xyz_right[2,:]!=0])!=0:
          mean_dis = points_xyz_right[2,points_xyz_right[2,:]!=0].mean()
          min_dis, max_dis = max(Z_min,mean_dis - 0.08), min(Z_max, mean_dis + 0.08)
          choose_right = ((points_xyz_right[2,:]>min_dis) & (points_xyz_right[2,:]<max_dis))
          choose_right = choose_right.flatten().nonzero()[0]
          tmpr = len(choose_right)
          if tmpr<80:
            print('what')
          if len(choose_right) < 10:
              choose_right = np.zeros((num_points), dtype=np.int64)
          elif len(choose_right) > num_points:
              c_mask = np.zeros(len(choose_right), dtype=int)
              c_mask[:num_points] = 1
              np.random.shuffle(c_mask)
              choose_right = choose_right[c_mask.nonzero()]
          else:
              choose_right = np.pad(choose_right, (0, num_points - len(choose_right)), 'wrap')
        else:
          choose_right = np.zeros((num_points), dtype=np.int64)   
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
      choose = np.stack([choose_left,choose_right])
      cloud = np.stack([hand_points_left,hand_points_right])
    else:
        choose = np.zeros((2,1024), dtype=np.int64)
        cloud = np.zeros((2,1024,3), dtype=np.float32)

    return choose, cloud

class PoseNetFeat(nn.Module):
    def __init__(self, opt):
        super(PoseNetFeat, self).__init__()
        self.opt = opt
        num_points = opt.SAMPLE_NUM
        self.sample_num_level2 = opt.sample_num_level2
        self.num_outputs = opt.PCA_SZ
        # output_channel of pointset
        channel_point = 3
        self.conv1 = torch.nn.Conv1d(channel_point, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        # output_channel of img_fmaps
        channel_img = 3
        self.e_conv1 = torch.nn.Conv1d(channel_img, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.ap1 = torch.nn.AvgPool1d(num_points)
        self.num_points = num_points
        self.netR_3 = nn.Sequential(
            # B*259*sample_num_level2*1
            nn.Conv2d(3+nstates_plus_2[2], nstates_plus_3[0], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[0]),
            nn.ReLU(inplace=True),
            # B*256*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[0], nstates_plus_3[1], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[1]),
            nn.ReLU(inplace=True),
            # B*512*sample_num_level2*1
            nn.Conv2d(nstates_plus_3[1], nstates_plus_3[2], kernel_size=(1, 1)),
            nn.BatchNorm2d(nstates_plus_3[2]),
            nn.ReLU(inplace=True),
            # B*1024*sample_num_level2*1
            nn.MaxPool2d((self.sample_num_level2,1),stride=1),
            # B*1024*1*1
        )
        
        self.netR_FC = nn.Sequential(
            # B*1024
            nn.Linear(nstates_plus_3[2], nstates_plus_3[3]),
            nn.BatchNorm1d(nstates_plus_3[3]),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(nstates_plus_3[3], nstates_plus_3[4]),
            nn.BatchNorm1d(nstates_plus_3[4]),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(nstates_plus_3[4], self.num_outputs),
            # B*num_outputs
        )        
    def forward(self, x, emb):
        x = F.relu(self.conv1(x))
        emb = F.relu(self.e_conv1(emb))
        pointfeat_1 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv2(x))
        emb = F.relu(self.e_conv2(emb))
        pointfeat_2 = torch.cat((x, emb), dim=1)

        x = F.relu(self.conv5(pointfeat_2))
        x = F.relu(self.conv6(x))

        ap_x = self.ap1(x) #128 + 256 + 1024

        # B*1024*1
        ap_x = ap_x.view(-1,1,nstates_plus_3[2])
        # B*1024
        # ap_x = self.netR_FC(ap_x)
        # B*num_outputs
        return ap_x 
    
class ResNetSimple(nn.Module):
    def __init__(self, model_type='resnet50',
                 pretrained=False,
                 fmapDim=[256, 256, 256, 256],
                 handNum=2,
                 heatmapDim=21,
                 opt=None):
        self.opt = opt
        self.global_feature_dim = 256 #512 * 1
        # batch_size = opt.batch_size // len(opt.gpus)
        # if 'params' in self.opt.heads:
        #     out_feature_size = self.opt.input_res // 8 
        #     init_pose_param = torch.zeros((batch_size, self.opt.heads['params'],out_feature_size,out_feature_size))
        #     self.register_buffer('mean_theta', init_pose_param.float())
        # if opt.iterations:
        #     self.iterations = 3
        # else:
        #     self.iterations = 1
            
        super(ResNetSimple, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=pretrained)
            self.expansion = 1
            self.p2 = nn.Conv2d(64, self.global_feature_dim, kernel_size=3, stride=1, padding=1)
            self.p3 = nn.ConvTranspose2d(128, self.global_feature_dim, kernel_size=4, stride=2, padding=1)
            self.p4 = nn.ConvTranspose2d(256, self.global_feature_dim, kernel_size=4, stride=4, padding=0)
            self.p5 = nn.ConvTranspose2d(512, self.global_feature_dim, kernel_size=8, stride=8, padding=0)

        elif model_type == 'resnet34':
            self.resnet = resnet34(pretrained=pretrained)
            self.expansion = 1
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=pretrained)
            self.expansion = 4
            self.p2 = nn.Conv2d(256, self.global_feature_dim, kernel_size=3, stride=1, padding=1)
            self.p3 = nn.ConvTranspose2d(512, self.global_feature_dim, kernel_size=4, stride=2, padding=1)
            self.p4 = nn.ConvTranspose2d(1024, self.global_feature_dim, kernel_size=4, stride=4, padding=0)
            self.p5 = nn.ConvTranspose2d(2048, self.global_feature_dim, kernel_size=8, stride=8, padding=0)

        elif model_type == 'resnet101':
            self.resnet = resnet101(pretrained=pretrained)
            self.expansion = 4
        elif model_type == 'resnet152':
            self.resnet = resnet152(pretrained=pretrained)
            self.expansion = 4
        self.p2_l2 = L2Norm(self.global_feature_dim, 10)
        self.p3_l2 = L2Norm(self.global_feature_dim, 10)
        self.p4_l2 = L2Norm(self.global_feature_dim, 10)
        self.p5_l2 = L2Norm(self.global_feature_dim, 10)
        self.feat = nn.Conv2d(self.global_feature_dim*4, self.global_feature_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_bn = nn.BatchNorm2d(self.global_feature_dim, momentum=0.01)
        self.feat_act = nn.ReLU(inplace=True)
        # self.pointnet = PoseNetFeat(opt)
        self.e_conv1 = nn.Conv2d(3, 3,kernel_size=3, stride=1, padding=1, bias=False)
        self.pointnet_plus = PointNet_Plus(opt)
        self.hms_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                                fDim=fmapDim,
                                                direction=['flat', 'up', 'up', 'up'], 
                                                out_dim=heatmapDim * handNum,up_scale = False) ## we want to keep hms 1/4 but mask the original.
        self.center_feat_up0 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.center_feat_up1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        # self.mano_head = nn.Linear(1024, 122)
        self.mano_head = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(256, 122),
            # B*num_outputs
        )
        fill_fc_weights(self.mano_head)
        self.joint_head_l = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(256, 22*3),
            # B*num_outputs
        )
        fill_fc_weights(self.joint_head_l)
        self.joint_head_r = nn.Sequential(
            # B*1024
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            # B*1024
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # B*512
            nn.Linear(256, 22*3),
            # B*num_outputs
        )
        fill_fc_weights(self.joint_head_r)
        # SFT_Net
        self.sft = SFTLayer(1024,1024)
        ### head decoders
        for head in sorted(self.opt.heads):
            num_output = self.opt.heads[head]
            # textures = _tranpose_and_gather_feat(output['texture'], batch['ind'])
            if 'params' in head and opt.iterations:
                extra_chanel = num_output
            else:
                extra_chanel = 0
            fc = nn.Sequential(
                nn.Conv2d(self.global_feature_dim + extra_chanel, 256,
                        kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, num_output,
                        kernel_size=1, stride=1, padding=0))

            if 'hm' in head or 'heatmaps' in head or 'handmap' in head:
                fc[-1].bias.data.fill_(-4.59)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
    
        for m in self.hms_decoder.modules():
            weights_init(m)

        self.dp_decoder = ResNetSimple_decoder(expansion=self.expansion,
                                               fDim=fmapDim,
                                               direction=['flat', 'up', 'up', 'up'],
                                               out_dim=handNum + 3 * 0,up_scale = True) # out_dim=handNum + 3 * handNum)
        self.handNum = handNum

        for m in self.dp_decoder.modules():
            weights_init(m)

    def forward(self, x, depth, ind, choose, cloud, K_new, valid):
        # if depth is not None:
        #     x = torch.cat([x,depth],1)
            # x = depth 
        point_wise_emb_l0 = F.relu(self.e_conv1(x)) #[b, 3, 256, 256] /512
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        point_wise_emb_l1 = x #[b, 64, 128, 128] /256
        x = self.resnet.maxpool(x)

        x4 = self.resnet.layer1(x) #[b, 256, 64, 64] /128
        p2 = self.p2(x4) #[b, 256, 64, 64] /128
        p2 = self.p2_l2(p2)
        x3 = self.resnet.layer2(x4)#[b, 512, 32, 32] /64
        p3 = self.p3(x3) #[b, 256, 64, 64] /128
        p3 = self.p3_l2(p3)
        x2 = self.resnet.layer3(x3)#[b, 1024, 16, 16] /32
        p4 = self.p4(x2) #[b, 256, 64, 64] /128
        p4 = self.p4_l2(p4)
        x1 = self.resnet.layer4(x2)#[b, 2048, 8, 8] /16
        p5 = self.p5(x1) #[b, 256, 64, 64] /128
        p5 = self.p5_l2(p5)

        if False:
            img_fmaps = [x1, x2, x3, x4]

            hms, hms_fmaps = self.hms_decoder(x1)
            out, dp_fmaps = self.dp_decoder(x1)
            mask = out[:, :self.handNum]
            dp = out[:, self.handNum:]
            return hms, mask, dp, None, \
                img_fmaps, hms_fmaps, dp_fmaps
        cat = torch.cat([p2, p3, p4, p5], dim=1) #[b, 1024, 64, 64] /128
        feat = self.feat(cat) #[b, 256, 64, 64]
        feat = self.feat_bn(feat)
        x0 = self.feat_act(feat)
        point_wise_emb_l2 = x0
        point_wise_emb = [point_wise_emb_l0, point_wise_emb_l1,point_wise_emb_l2] # 512/256/128


        ret = {}
        for head in self.opt.heads:
            if 'hm' in ret and ind is None:
                chms = ret['hm'].clone().detach()
                score = 0.5
                chms = _nms(chms, 5)
                K = int((chms[0] > score).float().sum())
                K = 1
                topk_scores, pred_ind_left, topk_ys, topk_xs = _topk(chms[:,:1,:,:], K)  
                topk_scores, pred_ind_right, topk_ys, topk_xs = _topk(chms[:,1:,:,:], K)     
                ind = torch.cat((pred_ind_left,pred_ind_right),dim=1)             
            # if 'params' in head:
            #     # do iterations for pose params
            #     thetas = []
            #     theta = self.mean_theta
            #     for _ in range(self.iterations):
            #         if self.opt.iterations:
            #             total_inputs = torch.cat([feat, theta], 1)
            #         else:
            #             total_inputs = feat
            #         theta = theta + self.__getattr__(head)(total_inputs)
            #         thetas.append(theta)
            #     ret[head] = thetas
            #     continue              
            ret[head] = self.__getattr__(head)(x0)

        hms, hms_fmaps = self.hms_decoder(x1)
        out, dp_fmaps = self.dp_decoder(x1)
        mask = out[:, :self.handNum]
        dp = None #out[:, self.handNum:]

        cnt = choose.sum() if choose is not None else 0
        if cnt == 0: # only test
            device = out.device
            choose_np, cloud_np = depth2pcl(depth, mask, K_new, valid)
            choose = torch.from_numpy(choose_np).unsqueeze(0).to(device)
            cloud = torch.from_numpy(cloud_np).unsqueeze(0).to(device)

        # here we add x0 refers to center-features.
        # img_fmaps = [x0, x1, x2, x3, x4]
        # img_fmaps = [x1, x2, x3, x4]
        # get center map and inds.
        x0_up0 = self.center_feat_up0(x0)
        x0_up1 = self.center_feat_up1(x0_up0)
        center_features = _tranpose_and_gather_feat(x0_up1, ind)
        # emb = x0.view(x0.shape[0], x0.shape[1], -1)
        # choose = ind.unsqueeze(1).repeat(1, x0.shape[1], 1)
        # center_features = torch.gather(emb, 2, choose).contiguous()

        # use DenseFusion code to fuse depth+img feature.
        # img_fmaps = [center_features, x2, x3, x4]
        # point_wise_emb_left = _tranpose_and_gather_feat(point_wise_emb_l0, choose[:,0,:]) #[b,1024,512]
        # point_wise_emb_right = _tranpose_and_gather_feat(point_wise_emb_l0, choose[:,1,:]) #cloud[:,0,:,:] # [b,1024,3]
        # estimation_left = self.pointnet(cloud[:,0,:,:].transpose(2, 1).contiguous(), point_wise_emb_left.transpose(2, 1).contiguous())
        # estimation_right = self.pointnet(cloud[:,1,:,:].transpose(2, 1).contiguous(), point_wise_emb_right.transpose(2, 1).contiguous())
        # fuse_feat = torch.cat((estimation_left,estimation_right),dim=1)
        # use Pointnet_plus code 
        fuse_left = self.pointnet_plus(cloud[:,0,:,:],point_wise_emb,choose[:,0,:])
        fuse_right = self.pointnet_plus(cloud[:,1,:,:],point_wise_emb,choose[:,1,:])
        fuse_feat = torch.cat((fuse_left,fuse_right),dim=1)
        # final_feat = torch.cat((center_features,fuse_feat),dim=2)
        fuse_feat = self.sft((fuse_feat.transpose(1,2).contiguous(),center_features))
        img_fmaps = [fuse_feat, x2, x3, x4]
        if False: # MANO branch
            ret['point2mano_left'] = self.mano_head(fuse_left.squeeze(1)).unsqueeze(1)
            ret['point2mano_right'] = self.mano_head(fuse_right.squeeze(1)).unsqueeze(1)
            ret['point2joint_left'] = self.joint_head_l(center_features[:,0,:].squeeze(1)).unsqueeze(1)
            ret['point2joint_right'] = self.joint_head_r(center_features[:,1,:].squeeze(1)).unsqueeze(1)
            ret['cnn2mano_left'] = _tranpose_and_gather_feat(ret['params'], ind[:,:1])
            ret['cnn2mano_right'] = _tranpose_and_gather_feat(ret['params'], ind[:,1:])        
        return hms, mask, dp, ret, \
            img_fmaps, hms_fmaps, dp_fmaps


class resnet_mid(nn.Module):
    def __init__(self,
                 model_type='resnet50',
                 in_fmapDim=[256, 256, 256, 256],
                 out_fmapDim=[256, 256, 256, 256]):
        super(resnet_mid, self).__init__()
        assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if model_type == 'resnet18' or model_type == 'resnet34':
            self.expansion = 1
        elif model_type == 'resnet50' or model_type == 'resnet101' or model_type == 'resnet152':
            self.expansion = 4

        self.img_fmaps_dim = [512 * self.expansion, 256 * self.expansion,
                              128 * self.expansion, 64 * self.expansion]
        self.dp_fmaps_dim = in_fmapDim
        self.hms_fmaps_dim = in_fmapDim

        self.convs = nn.ModuleList()
        for i in range(len(out_fmapDim)):
            inDim = self.dp_fmaps_dim[i] + self.hms_fmaps_dim[i]
            if i > 0:
                inDim = inDim + self.img_fmaps_dim[i]
            self.convs.append(conv1x1(inDim, out_fmapDim[i]))

        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
        )

        self.global_feature_dim = 512 *2 #self.expansion
        self.fmaps_dim = out_fmapDim

    def get_info(self):
        return {'global_feature_dim': self.global_feature_dim,
                'fmaps_dim': self.fmaps_dim}

    def forward(self, img_fmaps, hms_fmaps, dp_fmaps):
        if False:
            global_feature = self.output_layer(img_fmaps[0])
            fmaps = []
            for i in range(len(self.convs)):
                x = torch.cat((hms_fmaps[i], dp_fmaps[i]), dim=1)
                if i > 0:
                    x = torch.cat((x, img_fmaps[i]), dim=1)
                fmaps.append(self.convs[i](x))
            return global_feature, fmaps
        # change the flatten operation to convolution and 32,2048,8,8->32,2048,1
        # resize to [b,256,64,64]
        # global_feature = self.output_layer(img_fmaps[0])
        global_feature_left = img_fmaps[0][:,0,:]
        global_feature_right = img_fmaps[0][:,1,:]
        # global_feature_left = img_fmaps[0][:,:,0]
        # global_feature_right = img_fmaps[0][:,:,1]
        fmaps = []
        for i in range(len(self.convs)):
            x = torch.cat((hms_fmaps[i], dp_fmaps[i]), dim=1)
            if i > 0:
                x = torch.cat((x, img_fmaps[i]), dim=1)
            fmaps.append(self.convs[i](x))
        return global_feature_left, global_feature_right, fmaps
        # return global_feature,fmaps


# class HRnet_encoder(nn.Module):
#     def __init__(self, model_type, pretrained='', handNum=2, heatmapDim=21):
#         super(HRnet_encoder, self).__init__()
#         name = 'w' + model_type[model_type.find('hrnet') + 5:]
#         assert name in ['w18', 'w18_small_v1', 'w18_small_v2', 'w30', 'w32', 'w40', 'w44', 'w48', 'w64']

#         self.hrnet = get_hrnet(name=name,
#                                in_channels=3,
#                                head_type='none',
#                                pretrained='')

#         if os.path.isfile(pretrained):
#             print('load pretrained params: {}'.format(pretrained))
#             pretrained_dict = torch.load(pretrained)
#             model_dict = self.hrnet.state_dict()
#             pretrained_dict = {k: v for k, v in pretrained_dict.items()
#                                if k in model_dict.keys() and k.find('classifier') == -1}
#             model_dict.update(pretrained_dict)
#             self.hrnet.load_state_dict(model_dict)

#         self.fmaps_dim = list(self.hrnet.stage4_cfg['NUM_CHANNELS'])
#         self.fmaps_dim.reverse()

#         self.hms_decoder = self.mask_decoder(outDim=heatmapDim * handNum)
#         for m in self.hms_decoder.modules():
#             weights_init(m)

#         self.dp_decoder = self.mask_decoder(outDim=1 + 3 * handNum)
#         for m in self.dp_decoder.modules():
#             weights_init(m)

#     def mask_decoder(self, outDim=3):
#         last_inp_channels = 0
#         for temp in self.fmaps_dim:
#             last_inp_channels += temp

#         return nn.Sequential(
#             nn.Conv2d(
#                 in_channels=last_inp_channels, out_channels=last_inp_channels,
#                 kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(last_inp_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 in_channels=last_inp_channels, out_channels=outDim,
#                 kernel_size=1, stride=1, padding=0)
#         )

#     def forward(self, img):
#         ylist = self.hrnet(img)

#         # Upsampling
#         x0_h, x0_w = ylist[0].size(2), ylist[0].size(3)
#         x1 = F.interpolate(ylist[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
#         x2 = F.interpolate(ylist[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
#         x3 = F.interpolate(ylist[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
#         x = torch.cat([ylist[0], x1, x2, x3], 1)

#         hms = self.hms_decoder(x)
#         out = self.dp_decoder(x)
#         mask = out[:, 0]
#         dp = out[:, 1:]

#         ylist.reverse()
#         return hms, mask, dp, \
#             ylist, None, None


# class hrnet_mid(nn.Module):
#     def __init__(self,
#                  model_type,
#                  in_fmapDim=[256, 256, 256, 256],
#                  out_fmapDim=[256, 256, 256, 256]):
#         super(hrnet_mid, self).__init__()
#         name = 'w' + model_type[model_type.find('hrnet') + 5:]
#         assert name in ['w18', 'w18_small_v1', 'w18_small_v2', 'w30', 'w32', 'w40', 'w44', 'w48', 'w64']

#         self.convs = nn.ModuleList()
#         for i in range(len(out_fmapDim)):
#             self.convs.append(conv1x1(in_fmapDim[i], out_fmapDim[i]))

#         self.global_feature_dim = 2048
#         self.fmaps_dim = out_fmapDim

#         in_fmapDim.reverse()
#         self.incre_modules, self.downsamp_modules, \
#             self.final_layer = self._make_head(in_fmapDim)

#     def get_info(self):
#         return {'global_feature_dim': self.global_feature_dim,
#                 'fmaps_dim': self.fmaps_dim}

#     def _make_layer(self, block, inplanes, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(inplanes, planes * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion, momentum=0.1),
#             )

#         layers = []
#         layers.append(block(inplanes, planes, stride, downsample))
#         inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(inplanes, planes))

#         return nn.Sequential(*layers)

#     def _make_head(self, pre_stage_channels):
#         head_block = Bottleneck
#         head_channels = [32, 64, 128, 256]

#         # Increasing the #channels on each resolution
#         # from C, 2C, 4C, 8C to 128, 256, 512, 1024
#         incre_modules = []
#         for i, channels in enumerate(pre_stage_channels):
#             incre_module = self._make_layer(head_block,
#                                             channels,
#                                             head_channels[i],
#                                             1,
#                                             stride=1)
#             incre_modules.append(incre_module)
#         incre_modules = nn.ModuleList(incre_modules)

#         # downsampling modules
#         downsamp_modules = []
#         for i in range(len(pre_stage_channels) - 1):
#             in_channels = head_channels[i] * head_block.expansion
#             out_channels = head_channels[i + 1] * head_block.expansion

#             downsamp_module = nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels,
#                           out_channels=out_channels,
#                           kernel_size=3,
#                           stride=2,
#                           padding=1),
#                 nn.BatchNorm2d(out_channels, momentum=0.1),
#                 nn.ReLU(inplace=True)
#             )

#             downsamp_modules.append(downsamp_module)
#         downsamp_modules = nn.ModuleList(downsamp_modules)

#         final_layer = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=head_channels[3] * head_block.expansion,
#                 out_channels=2048,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0
#             ),
#             nn.BatchNorm2d(2048, momentum=0.1),
#             nn.ReLU(inplace=True)
#         )

#         return incre_modules, downsamp_modules, final_layer

#     def forward(self, img_fmaps, hms_fmaps=None, dp_fmaps=None):
#         fmaps = []
#         for i in range(len(self.convs)):
#             fmaps.append(self.convs[i](img_fmaps[i]))

#         img_fmaps.reverse()
#         y = self.incre_modules[0](img_fmaps[0])
#         for i in range(len(self.downsamp_modules)):
#             y = self.incre_modules[i + 1](img_fmaps[i + 1]) + \
#                 self.downsamp_modules[i](y)

#         y = self.final_layer(y)

#         if torch._C._get_tracing_state():
#             y = y.flatten(start_dim=2).mean(dim=2)
#         else:
#             y = F.avg_pool2d(y, kernel_size=y.size()
#                              [2:]).view(y.size(0), -1)

#         return y, fmaps


def load_encoder(opt):
    if opt.depth:
        predtrain_tag = False
    else:
        predtrain_tag = True
    encoder = ResNetSimple(model_type='resnet50',
                            pretrained=predtrain_tag,
                            fmapDim=[128, 128, 128, 128],
                            handNum=2,
                            heatmapDim=21,
                            opt=opt)
    mid_model = resnet_mid(model_type='resnet50',
                            in_fmapDim=[128, 128, 128, 128],
                            out_fmapDim=opt.DECONV_DIMS)
    # if cfg.MODEL.ENCODER_TYPE.find('hrnet') != -1:
    #     encoder = HRnet_encoder(model_type=cfg.MODEL.ENCODER_TYPE,
    #                             pretrained=cfg.MODEL.ENCODER_PRETRAIN_PATH,
    #                             handNum=2,
    #                             heatmapDim=21)
    #     mid_model = hrnet_mid(model_type=cfg.MODEL.ENCODER_TYPE,
    #                           in_fmapDim=encoder.fmaps_dim,
    #                           out_fmapDim=cfg.MODEL.DECONV_DIMS)

    return encoder, mid_model
