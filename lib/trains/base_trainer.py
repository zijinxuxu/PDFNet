from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import torch
from progress.bar import Bar

from models.data_parallel import DataParallel
from lib.utils.utils import AverageMeter
import cv2
import numpy as np
import os
import json
import pickle
from lib.utils.eval import main as eval_main
from lib.utils.eval import align_w_scale 
from lib.models.utils import _sigmoid, _tranpose_and_gather_feat
from lib.utils.image import get_affine_transform, affine_transform, affine_transform_array
from torch.nn.parallel import DistributedDataParallel as DDP

# This class can be removed
class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch, mode, epoch):
        # address IntagHand-GCN 
        if True:
            if mode == 'train':
                ind = batch['ind']
            else:
                ind = None
            if False: # only test for MANO branch
                ret = self.model(batch['input'], batch['choose'], batch['cloud'], batch['depth'], ind, batch['K_new'], batch['valid'])
                if mode == 'train':
                    loss, loss_stats, rendered, masks = self.loss.origforward(ret, mode, batch,epoch)
                elif mode == 'val' or mode == 'test':
                    vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, vertex_pred_off, joints_pred_off, vertex_gt_off, joints_gt_off = self.loss.origforward(ret, mode, batch,epoch)
                    return vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, vertex_pred_off, joints_pred_off, vertex_gt_off, joints_gt_off

            else:    
                if 'depth' in batch:
                    result, paramsDict, handDictList, otherInfo = self.model(batch['input'], batch['choose'], batch['cloud'], batch['depth'], ind, batch['K_new'], batch['valid'])
                else:
                    result, paramsDict, handDictList, otherInfo = self.model(batch['input'], None, None, None, ind, None, None)            
                # result: GCN upsample(778,3); handDictList: GCN out(252,3); otherinfo: GCN_to_ MANO out(778,3), hms(42), mask(2), dense(6), ret_hm(2) ret_wh(2) ret_hm(122)
                # result, paramsDict, handDictList, otherInfo = self.model(batch['input'])

                if mode == 'train':
                    loss, loss_stats, rendered, masks = self.loss(result, paramsDict, handDictList, otherInfo, batch, mode,epoch)
                elif mode == 'val' or mode == 'test':
                    vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, vertex_pred_off, joints_pred_off, vertex_gt_off, joints_gt_off = self.loss(result, paramsDict, handDictList, otherInfo, batch, mode,epoch)
                    return vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, vertex_pred_off, joints_pred_off, vertex_gt_off, joints_gt_off
        else: # MANO branch
            if mode == 'train':
                if 'heatmaps' in batch:
                    tmp_heatmaps = batch['heatmaps']
                else:
                    tmp_heatmaps = None
                if 'depth' in batch:
                    outputs, ind = self.model(batch['input'], tmp_heatmaps, batch['depth'], batch['ind'])
                else:
                    outputs, ind = self.model(batch['input'], tmp_heatmaps, None, batch['ind'])
                loss, loss_stats, rendered, masks = self.loss.origforward(outputs, mode, batch)
            elif mode == 'val' or mode == 'test':
                # loss, loss_stats, rendered, masks = self.test(outputs, batch)
                if 'depth' in batch:
                    outputs, ind = self.model(batch['input'], None, batch['depth'], None)
                else:
                    outputs, ind = self.model(batch['input'], None, None, None)
                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred = self.loss.origforward(outputs, mode, batch)
                return vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred

        return loss, loss_stats, rendered, masks


class BaseTrainer(object):
    def __init__(
            self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        if optimizer:
            self.loss_stats, self.loss = self._get_losses(opt)
            self.model_with_loss = ModleWithLoss(model, self.loss)
        self.model = model
        self.optimizer.add_param_group({'params': self.loss.parameters()})

    def set_device(self, gpus, chunk_sizes, device, local_rank):
        if local_rank is not None:
            self.model_with_loss = DDP(
                self.model_with_loss.cuda(), device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
        else:
            if len(gpus) >= 1:
                if self.optimizer:
                    self.model_with_loss = DataParallel(
                        self.model_with_loss, device_ids=gpus,
                        chunk_sizes=chunk_sizes).to(device)
            else:
                if self.optimizer:
                    self.model_with_loss = self.model_with_loss.to(device)
                self.model = self.model.to(device)

    def run_epoch(self, phase, epoch, data_loader, logger=None):
      pass


    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader, logger=None):
        return self.run_epoch('train', epoch, data_loader, logger=logger)

    def evaluation(self, eval_loader, logger=None):
      pass


    def val(self, test_loader, logger=None):
      pass
