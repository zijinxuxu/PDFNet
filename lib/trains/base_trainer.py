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
        model_with_loss = self.model_with_loss
        model_with_loss.train()

        torch.cuda.empty_cache()

        opt = self.opt

        results = {}
        data_time, batch_time, ema_time = AverageMeter(), AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = min(len(data_loader), 10000) if opt.num_iters < 0 else opt.num_iters
        num_iters = len(data_loader)
        if opt.local_rank ==0: # only write once
            bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
        end = time.time()

        # gradual_lr
        if opt.gradual_lr:
            opt.reproj_weight /= pow(10, 0.02)
            opt.photometric_weight *= pow(10, 0.02)

        for iter_id, batch in enumerate(data_loader):
            if iter_id >= num_iters:
                break
            data_time.update(time.time() - end)

            for k in batch:
                if k != 'meta':# and k != 'dataset':
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            loss, loss_stats, rendered, gpu_mask = model_with_loss(batch,'train',epoch)

            if opt.photometric_loss: # finetune using hard mode sample
                valid_loss, idxs = torch.topk(loss, int(0.7 * loss.size()[0]))    
                loss = torch.mean(valid_loss)
            else:
                loss = loss.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if opt.local_rank ==0: # only write once
                Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                    epoch, iter_id, num_iters, phase=phase,
                    total=bar.elapsed_td, eta=bar.eta_td)
                for l in avg_loss_stats:
                    avg_loss_stats[l].update(
                        loss_stats[l].mean().item(), batch['input'].size(0))
                    Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
                    Bar.suffix = Bar.suffix + '|cur_{} {:.4f} '.format(l, avg_loss_stats[l].val)
                if opt.print_iter > 0:
                    if iter_id % opt.print_iter == 0:
                        print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix))
                else:
                    bar.next()

                ## TensorboardX
                step = (epoch - 1) * num_iters + iter_id
                if logger is not None:
                    if step % 10 == 0:
                        for k, v in avg_loss_stats.items():
                            logger.scalar_summary('train_{}'.format(k), v.avg, step)
                    if step % 500 == 0 and self.opt.photometric_loss:
                        # print('logger: {}'.format(step))
                        img, chosen_img, gt_img = [], [], []
                        img.append(batch['image'][batch['mask'].sum(dim=1)>0].cpu())
                        if rendered is not None:
                            chosen_img.append(rendered.detach().cpu())
                            if 'Vs' in batch:
                                rendered_gt, rendered_gt_mask = self.render(batch['Vs'], batch['Ts'])
                                gt_img.append(rendered_gt.detach().cpu())
                            img = torch.cat(img, 0)
                            chosen_img = torch.clamp(torch.cat(chosen_img, 0), 0., 1.)
                            if len(gt_img) != 0:
                                gt_img = torch.clamp(torch.cat(gt_img, 0), 0., 1.)
                                t = torch.cat([img, chosen_img, gt_img], 2).permute(0, 3, 1, 2).contiguous()
                            else:
                                t = torch.cat([img, chosen_img], 2).permute(0, 3, 1, 2).contiguous()
                            logger.image_summary('train', t[:4], step)

        if opt.local_rank ==0: # only write once
            bar.finish()
            ret = {k: v.avg for k, v in avg_loss_stats.items()}
            ret['time'] = bar.elapsed_td.total_seconds() / 60.

            return ret, results
        else:
            return None, None

    def _get_losses(self, opt):
        raise NotImplementedError

    def train(self, epoch, data_loader, logger=None):
        return self.run_epoch('train', epoch, data_loader, logger=logger)

    def evaluation(self, eval_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()
        if isinstance(model_with_loss, DDP):
            model_with_loss = model_with_loss.module
            
        torch.cuda.empty_cache()
        xyz_pred_list, verts_pred_list = list(), list()
        xyz_gt_list, verts_gt_list = list(), list()
        H2O_list = {"modality": "RGBD"}
        local_list = {}
        bar = Bar("TEST", max=len(eval_loader))

        hand_num = 2 # or 2. modified according to your model.
        lmpjpe = [[] for _ in range(21*1)] # treat right and left hand identical 
        rmpjpe = [[] for _ in range(21*1)]
        lmpjpe_off = [[] for _ in range(21*1)] # treat right and left hand identical 
        rmpjpe_off = [[] for _ in range(21*1)]
        mpix = [[] for _ in range(21*hand_num)]  
        mpvpe = [[] for _ in range(778)] # treat right and left hand identical   
        action_id = 1
        left_joints_loss_all, right_joints_loss_all = 0, 0
        left_verts_loss_all, right_verts_loss_all = 0, 0
        left_joints_loss_all_off, right_joints_loss_all_off = 0, 0
        left_verts_loss_all_off, right_verts_loss_all_off = 0, 0
        lms_loss_all = 0
        with torch.no_grad():
            for step, data in enumerate(eval_loader):
                for k in data:
                    if k != 'meta':# and k != 'dataset':
                        data[k] = data[k].to(device=self.opt.device, non_blocking=True)

                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred, vertex_pred_off, joints_pred_off, vertex_gt_off, joints_gt_off  = model_with_loss(data,'test',None)         

                if self.opt.dataset == 'H2O':
                    # imgTensors = data[0].cuda()
                    joints_left_gt = joints_gt[:,0,:,:]
                    verts_left_gt = vertex_gt[:,0,:,:] if vertex_gt is not None else None
                    joints_right_gt = joints_gt[:,1,:,:]
                    verts_right_gt = vertex_gt[:,1,:,:] if vertex_gt is not None else None
                    lms21_left_gt = data['lms_left_gt'][:,:,:]
                    lms21_right_gt = data['lms_right_gt'][:,:,:]

                    if False:
                        img = (np.squeeze(data['image'][0])).detach().cpu().numpy().astype(np.float32)
                        cv2.imwrite('img_orig.jpg',img)

                    # 2. use otherInfo['Manolist] verts
                    verts_left_pred =  vertex_pred[:,0,:,:] if vertex_pred is not None else None
                    verts_right_pred = vertex_pred[:,1,:,:] if vertex_pred is not None else None
                    joints_left_pred =  joints_pred[:,0,:,:] if joints_pred is not None else None
                    joints_right_pred = joints_pred[:,1,:,:] if joints_pred is not None else None
                    lms21_left_pred = lms21_pred[:,0,:,:]
                    lms21_right_pred = lms21_pred[:,1,:,:]

                    if joints_gt is not None and joints_pred is not None:
                        joint_left_loss = torch.norm((joints_left_pred - joints_left_gt), dim=-1)
                        joint_left_loss = joint_left_loss.detach().cpu().numpy()

                        joint_right_loss = torch.norm((joints_right_pred - joints_right_gt), dim=-1)
                        joint_right_loss = joint_right_loss.detach().cpu().numpy()

                    if vertex_gt is not None and vertex_pred is not None:
                        vert_left_loss = torch.norm((verts_left_pred - verts_left_gt), dim=-1)
                        vert_left_loss = vert_left_loss.detach().cpu().numpy()

                        vert_right_loss = torch.norm((verts_right_pred - verts_right_gt), dim=-1) 
                        vert_right_loss = vert_right_loss.detach().cpu().numpy()

                    lms_left_loss = torch.norm((lms21_left_pred -lms21_left_gt), dim=-1).detach().cpu().numpy()
                    lms_right_loss = torch.norm((lms21_right_pred -lms21_right_gt), dim=-1).detach().cpu().numpy()

                    lms_loss_all = lms_loss_all + (lms_left_loss + lms_right_loss).mean()/2
                    if joints_gt is not None and joints_pred is not None:
                        left_joints_loss_all = left_joints_loss_all + joint_left_loss.mean()*1000
                        right_joints_loss_all = right_joints_loss_all + joint_right_loss.mean()*1000
                    if vertex_gt is not None and vertex_pred is not None:
                        left_verts_loss_all = left_verts_loss_all  + vert_left_loss.mean()*1000
                        right_verts_loss_all = right_verts_loss_all + vert_right_loss.mean()*1000

                    # again, calculate joint_off
                    joints_left_gt = joints_gt_off[:,0,:,:]
                    verts_left_gt = vertex_gt_off[:,0,:,:] if vertex_gt_off is not None else None
                    joints_right_gt = joints_gt_off[:,1,:,:]
                    verts_right_gt = vertex_gt_off[:,1,:,:] if vertex_gt_off is not None else None

                    if False:
                        img = (np.squeeze(data['image'][0])).detach().cpu().numpy().astype(np.float32)
                        cv2.imwrite('img_orig.jpg',img)

                    # 2. use otherInfo['Manolist] verts
                    verts_left_pred =  vertex_pred_off[:,0,:,:]
                    verts_right_pred = vertex_pred_off[:,1,:,:]
                    joints_left_pred =  joints_pred_off[:,0,:,:]
                    joints_right_pred = joints_pred_off[:,1,:,:]

                    joint_left_loss = torch.norm((joints_left_pred - joints_left_gt), dim=-1)
                    joint_left_loss = joint_left_loss.detach().cpu().numpy()

                    joint_right_loss = torch.norm((joints_right_pred - joints_right_gt), dim=-1)
                    joint_right_loss = joint_right_loss.detach().cpu().numpy()

                    if vertex_gt is not None:
                        vert_left_loss = torch.norm((verts_left_pred - verts_left_gt), dim=-1)
                        vert_left_loss = vert_left_loss.detach().cpu().numpy()

                        vert_right_loss = torch.norm((verts_right_pred - verts_right_gt), dim=-1)
                        vert_right_loss = vert_right_loss.detach().cpu().numpy()

                    lms_left_loss = torch.norm((lms21_left_pred -lms21_left_gt), dim=-1).detach().cpu().numpy()
                    lms_right_loss = torch.norm((lms21_right_pred -lms21_right_gt), dim=-1).detach().cpu().numpy()

                    left_joints_loss_all_off = left_joints_loss_all_off + joint_left_loss.mean()*1000
                    right_joints_loss_all_off = right_joints_loss_all_off + joint_right_loss.mean()*1000
                    if vertex_gt is not None:
                        left_verts_loss_all_off = left_verts_loss_all_off  + vert_left_loss.mean()*1000
                        right_verts_loss_all_off = right_verts_loss_all_off + vert_right_loss.mean()*1000

                    # bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                    # bar.next()
                    # continue        
                    if self.opt.batch_size == 1:
                        if data['id'][0] == action_id + 1:
                            H2O_list.update({'{}'.format(action_id): local_list})
                            action_id = action_id + 1
                            local_list = {}

                        frame_num = data['frame_num'][0]
                        local_list.update({'{:06d}.txt'.format(frame_num):joints_pred.reshape(-1).tolist()})

                if False:                         
                    lms_gt = data['lms'].view_as(lms21_pred)
                    if joints_gt is None:
                        joints_gt = data['joints'] 

                    for i in range(lms_gt.shape[0]):
                        tmp_lms_gt = (lms_gt)[i].reshape(-1,2).cpu().numpy()
                        tmp_lms_pred = (lms21_pred)[i].reshape(-1,2).cpu().numpy()
                        tmp_joints_pred = (joints_pred)[i].reshape(-1,3).cpu().numpy() if joints_pred is not None else None
                        tmp_verts_pred = vertex_pred[i].reshape(-1,3).cpu().numpy() if vertex_pred is not None else None
                        tmp_joints_gt = joints_gt[i].reshape(-1,3).cpu().numpy()
                        tmp_joints_pred_off = (joints_pred_off)[i].reshape(-1,3).cpu().numpy()
                        tmp_joints_gt_off = joints_gt_off[i].reshape(-1,3).cpu().numpy()
                        # tmp_xyz_gt = xyz_gt[i][0].reshape(-1,3).cpu().numpy()
                        # tmp_verts_gt = vertex_gt[i].reshape(-1,3).cpu().numpy()
                        # index_root_bone_length = np.sqrt(np.sum((tmp_joints_pred[10, :] - tmp_joints_pred[9, :])**2))
                        # gt_bone_length = np.sqrt(np.sum((tmp_joints_gt[10, :] - tmp_joints_gt[9, :])**2))
                        # xyz_pred_aligned = (tmp_joints_pred - tmp_joints_pred[9,:])/index_root_bone_length*gt_bone_length*1000
                        # verts_pred_aligned = (tmp_verts_pred - tmp_verts_pred[9,:])/index_root_bone_length*gt_bone_length*1000 

                        if tmp_joints_pred_off.shape[0] > 21 and tmp_joints_gt_off.shape[0] > 21:
                            joints_gt_aligned = tmp_joints_gt.copy()*1000 if tmp_joints_gt is not None else None
                            xyz_pred_aligned = tmp_joints_pred.copy()*1000 if tmp_joints_pred is not None else None
                            joints_gt_aligned_off = tmp_joints_gt_off.copy()*1000
                            xyz_pred_aligned_off = tmp_joints_pred_off.copy()*1000
                        else:
                            joints_gt_aligned = (tmp_joints_gt - tmp_joints_gt[9,:])*1000
                        
                        # verts_gt_aligned = (tmp_verts_gt - tmp_verts_gt[0,:])*1000
                        # select one hand to align for InterHand
                        if self.opt.dataset == 'RHD' and self.opt.task == 'interact':
                            select = int(data['select'][0].cpu())
                            # if select == 0:
                            #     continue # jump left hand for test
                            hand_num = 1
                            xyz_pred_aligned = xyz_pred_aligned[select*21:21+select*21,:].copy()
                            joints_gt_aligned = joints_gt_aligned[select*21:21+select*21,:].copy()
                            xyz_pred_aligned_off = xyz_pred_aligned_off[select*21:21+select*21,:].copy()
                            joints_gt_aligned_off = joints_gt_aligned_off[select*21:21+select*21,:].copy()
                            gt_length = np.sqrt(np.sum((joints_gt_aligned[9] - joints_gt_aligned[0])**2))
                            pred_length = np.sqrt(np.sum((xyz_pred_aligned[9] - xyz_pred_aligned[0])**2))
                            # xyz_pred_aligned_off = xyz_pred_aligned_off * gt_length / pred_length
                            tmp_lms_pred = tmp_lms_pred[select*21:21+select*21,:].copy()
                            tmp_lms_gt = tmp_lms_gt[select*21:21+select*21,:].copy()
                        for j in range(tmp_lms_pred.shape[0]):
                            if tmp_lms_gt[j][0] == 0:
                                continue # remove outliers    
                            mpix[j].append(np.sqrt(np.sum((tmp_lms_pred[j] - tmp_lms_gt[j])**2)))
                            if j < 21:             
                                if xyz_pred_aligned is not None:
                                    lmpjpe[j].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                                lmpjpe_off[j].append(np.sqrt(np.sum((xyz_pred_aligned_off[j] - joints_gt_aligned_off[j])**2)))
                            else:
                                if xyz_pred_aligned is not None:
                                    rmpjpe[j-21].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                                rmpjpe_off[j-21].append(np.sqrt(np.sum((xyz_pred_aligned_off[j] - joints_gt_aligned_off[j])**2)))
                    
                        # for j in range(778):
                        #     mpvpe[j].append(np.sqrt(np.sum((verts_pred_aligned[j] - verts_gt_aligned[j])**2)))
                        if False:
                            verts_gt_aligned = verts_gt[i][0].reshape(-1,3).cpu().numpy()*1000
                            eval_main(joints_gt_aligned,verts_gt_aligned,xyz_pred_aligned,verts_pred_aligned,'./')                        
                # if args.phase == 'eval':
                #     save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(eval_loader))
                bar.next()
        bar.finish()

        eval_summary = 'MPJPE for each joint: \n'
        score_path = os.path.join(self.opt.root_dir, 'H2O-val.txt')   
        if self.opt.dataset == 'H2O':
            print('abs')  
            print(lms_loss_all/len(eval_loader))
            print(left_joints_loss_all/len(eval_loader))
            print(left_verts_loss_all/len(eval_loader))
            print(right_joints_loss_all/len(eval_loader))
            print(right_verts_loss_all/len(eval_loader))   
            print('off')   
            print(left_joints_loss_all_off/len(eval_loader))
            print(left_verts_loss_all_off/len(eval_loader))
            print(right_joints_loss_all_off/len(eval_loader))
            print(right_verts_loss_all_off/len(eval_loader)) 
            with open(score_path, 'a') as fo:
                fo.write('eval \n')
                fo.write('abs_left_joints_loss_all: %.2f\n' % (left_joints_loss_all/len(eval_loader)))
                fo.write('abs_right_joints_loss_all: %.2f\n' % (right_joints_loss_all/len(eval_loader)))
                fo.write('abs_left_verts_loss_all: %.2f\n' % (left_verts_loss_all/len(eval_loader)))
                fo.write('abs_right_verts_loss_all: %.2f\n' % (right_verts_loss_all/len(eval_loader)))
                fo.write('off_left_joints_loss_all: %.2f\n' % (left_joints_loss_all_off/len(eval_loader)))
                fo.write('off_right_joints_loss_all: %.2f\n' % (right_joints_loss_all_off/len(eval_loader)))
                fo.write('off_left_verts_loss_all: %.2f\n' % (left_verts_loss_all_off/len(eval_loader)))
                fo.write('off_right_verts_loss_all: %.2f\n' % (right_verts_loss_all_off/len(eval_loader)))
            # return None, None
            # append the last term.
            if self.opt.batch_size == 1:
                H2O_list.update({'{}'.format(action_id): local_list})

        if False: # test for evaluation score
            eval_main(xyz_gt_list,verts_gt_list,xyz_pred_list,verts_pred_list,'./')  
            return None, None                      

        if joints_gt is None:
            for j in range(21*hand_num):
                mpix[j] = np.mean(np.stack(mpix[j]))
                # joint_name = self.skeleton[j]['name']
                print('lms_{0}:{1}'.format(j,mpix[j])) 
            # print(eval_summary)
            print('MPJPE_lms: %.2f' % (np.mean(mpix[:63])))
            with open(score_path, 'a') as fo:
                fo.write('UV_mean2d: %f\n' % np.mean(mpix[:63]))
            print('Scores written to: %s' % score_path)

            return None, None

        # for j in range(21*hand_num):
        #     mpix[j] = np.mean(np.stack(mpix[j]))
        #     print('lms_{0}:{1}'.format(j,mpix[j])) 
        # for j in range(21*1):
        #     if self.opt.dataset == 'H2O':
        #         rmpjpe[j] = np.mean(np.stack(rmpjpe[j]))
        #         print('rjoint_{0}:{1}'.format(j,rmpjpe[j])) 
        #         rmpjpe_off[j] = np.mean(np.stack(rmpjpe_off[j]))
        #         print('rjoint__off_{0}:{1}'.format(j,rmpjpe_off[j])) 

        #     lmpjpe[j] = np.mean(np.stack(lmpjpe[j]))
        #     print('ljoint_{0}:{1}'.format(j,lmpjpe[j])) 
        #     lmpjpe_off[j] = np.mean(np.stack(lmpjpe_off[j]))
        #     print('ljoint__off_{0}:{1}'.format(j,lmpjpe_off[j])) 
        # # print(eval_summary)
        # if self.opt.dataset == 'RHD':
        #     print('MPJPE_joint: %.2f:' % (np.mean(lmpjpe[:])))
        #     print('MPJPE_lms: %.2f' % (np.mean(mpix[:21])))
        #     with open(score_path, 'a') as fo:
        #         fo.write('UV_mean2d: %f\n' % np.mean(mpix[:21]))
        #         fo.write('UV_mean3d_left: %f\n, Off_UV_mean3d_left: %f\n' % (np.mean(lmpjpe[:]),np.mean(lmpjpe_off[:])))
        #     return None, None
        # print('MPJPE_ljoint: %.2f, MPJPE_rjoint: %.2f' % (np.mean(lmpjpe[:]),np.mean(rmpjpe[:])))
        # print('MPJPE_lms: %.2f' % (np.mean(mpix[:42])))
        # with open(score_path, 'a') as fo:
        #     fo.write('UV_mean2d: %f\n' % np.mean(mpix[:42]))
        #     fo.write('UV_mean3d_left: %f\n, UV_mean3d_right: %f\n' % (np.mean(lmpjpe[:]),np.mean(rmpjpe[:])))
        #     fo.write('Off_UV_mean3d_left: %f\n, Off_UV_mean3d_right: %f\n' % (np.mean(lmpjpe_off[:]),np.mean(rmpjpe_off[:])))
        # print('Scores written to: %s' % score_path)

        ### save to json file for submitting.
        # xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        # verts_pred_list = [x.tolist() for x in verts_pred_list]
        ### submit to server.
        if self.opt.batch_size == 1:
            with open(os.path.join(self.opt.root_dir, 'hand_poses' + '.json'), 'w') as fo:
                json.dump(H2O_list, fo)
            print('Save json file at ' + os.path.join(self.opt.root_dir, 'hand_poses' + '.json'))
    
        return None, None

    def val(self, test_loader, logger=None):
        model_with_loss = self.model_with_loss
        model_with_loss.eval()

        torch.cuda.empty_cache()
        xyz_pred_list, verts_pred_list = list(), list()
        xyz_gt_list, verts_gt_list = list(), list()
        bar = Bar("TEST", max=len(test_loader))

        hand_num = 2 # or 2. modified according to your model.
        mpjpe = [[] for _ in range(21*hand_num)] # treat right and left hand identical 
        mpix = [[] for _ in range(21*hand_num)]   
        mpvpe = [[] for _ in range(778)] # treat right and left hand identical   
        with torch.no_grad():
            for step, data in enumerate(test_loader):
                for k in data:
                    if k != 'meta':# and k != 'dataset':
                        data[k] = data[k].to(device=self.opt.device, non_blocking=True)

                vertex_pred, joints_pred, vertex_gt, joints_gt, lms21_pred  = model_with_loss(data,'val',None)    

                # gt_hand_type = data['hand_type'][0][0]
                if 'lms' in data:
                    lms_gt = data['lms'].view_as(lms21_pred)
                else:
                    lms_gt = torch.cat((data['lms_left_gt'],data['lms_right_gt'])).view_as(lms21_pred)
                if joints_gt is None:
                    joints_gt = data['joints'] 
                                              
                for i in range(lms_gt.shape[0]):
                    tmp_lms_gt = (lms_gt)[i].reshape(-1,2).cpu().numpy()
                    tmp_lms_pred = (lms21_pred)[i].reshape(-1,2).cpu().numpy()
                    tmp_joints_pred = (joints_pred)[i].reshape(-1,3).cpu().numpy()
                    # tmp_verts_pred = vertex_pred[i].reshape(-1,3).cpu().numpy()
                    tmp_joints_gt = joints_gt[i].reshape(-1,3).cpu().numpy()
                    # tmp_xyz_gt = xyz_gt[i][0].reshape(-1,3).cpu().numpy()
                    # tmp_verts_gt = vertex_gt[i].reshape(-1,3).cpu().numpy()
                    index_root_bone_length = np.sqrt(np.sum((tmp_joints_pred[10, :] - tmp_joints_pred[9, :])**2))
                    gt_bone_length = np.sqrt(np.sum((tmp_joints_gt[10, :] - tmp_joints_gt[9, :])**2))
                    # xyz_pred_aligned = (tmp_joints_pred - tmp_joints_pred[9,:])/index_root_bone_length*gt_bone_length*1000
                    # verts_pred_aligned = (tmp_verts_pred - tmp_verts_pred[9,:])/index_root_bone_length*gt_bone_length*1000 

                    if tmp_joints_pred.shape[0] > 21 and tmp_joints_gt.shape[0] > 21:
                        # here we consider RHD two hand case
                        # for j in range(len(data['valid'][i])):
                        #     if data['valid'][i][j] == 0: # no visible hand in left/right
                        #         tmp_joints_gt[j*21:21+j*21,:] = tmp_joints_gt[j*21:21+j*21,:] * 0
                        #         tmp_joints_pred[j*21:21+j*21,:] = tmp_joints_pred[j*21:21+j*21,:] * 0
                        #     else:
                        #         tmp_joints_gt[j*21:21+j*21,:] = (tmp_joints_gt[j*21:21+j*21,:] -tmp_joints_gt[9+j*21,:])*1000
                        #         tmp_joints_pred[j*21:21+j*21,:] = (tmp_joints_pred[j*21:21+j*21,:] -tmp_joints_pred[9+j*21,:])*1000
                        joints_gt_aligned = tmp_joints_gt.copy()*1000
                        xyz_pred_aligned = tmp_joints_pred.copy()*1000
                    else:
                        joints_gt_aligned = (tmp_joints_gt - tmp_joints_gt[9,:])*1000
                    
                    # verts_gt_aligned = (tmp_verts_gt - tmp_verts_gt[0,:])*1000
                    # select one hand to align for InterHand
                    if self.opt.dataset == 'InterHand' and self.opt.task == 'interact':
                        select = int(data['handtype'][0][0].cpu())
                        # if select == 0:
                        #     continue # jump left hand for test
                        hand_num = 1
                        xyz_pred_aligned = xyz_pred_aligned[select*21:21+select*21,:].copy()
                        joints_gt_aligned = joints_gt_aligned[select*21:21+select*21,:].copy()
                        tmp_lms_pred = tmp_lms_pred[select*21:21+select*21,:].copy()
                        tmp_lms_gt = tmp_lms_gt[select*21:21+select*21,:].copy()
                    # xyz_pred_aligned = align_w_scale(joints_gt_aligned, xyz_pred_aligned) 
                    # xyz_pred_aligned[:21,:] = align_w_scale(joints_gt_aligned[:21,:], xyz_pred_aligned[:21,:]) 
                    # xyz_pred_aligned[21:,:] = align_w_scale(joints_gt_aligned[21:,:], xyz_pred_aligned[21:,:]) 
                    # print('R, s, s1, s2, t1, t2',align_w_scale(joints_gt_aligned, xyz_pred_aligned, True))
                    for j in range(tmp_lms_pred.shape[0]):
                        if tmp_lms_gt[j][0] == 0:
                            continue # remove outliers                        
                        mpix[j].append(np.sqrt(np.sum((tmp_lms_pred[j] - tmp_lms_gt[j])**2)))
                        mpjpe[j].append(np.sqrt(np.sum((xyz_pred_aligned[j] - joints_gt_aligned[j])**2)))
                  
                    # for j in range(778):
                    #     mpvpe[j].append(np.sqrt(np.sum((verts_pred_aligned[j] - verts_gt_aligned[j])**2)))
                    if False:
                        verts_gt_aligned = verts_gt[i][0].reshape(-1,3).cpu().numpy()*1000
                        eval_main(joints_gt_aligned,verts_gt_aligned,xyz_pred_aligned,verts_pred_aligned,'./')                        
                # if args.phase == 'eval':
                #     save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
                bar.suffix = '({batchs}/{size})' .format(batchs=step+1, size=len(test_loader))
                bar.next()
        bar.finish()

        if False: # test for evaluation score
            eval_main(xyz_gt_list,verts_gt_list,xyz_pred_list,verts_pred_list,'./')  
            return None, None                      

        eval_summary = 'MPJPE for each joint: \n'
        score_path = os.path.join(self.opt.root_dir, 'H2O-val.txt')   
        if self.opt.task == 'artificial': # there may be less than 10 hands in artificial and got error
            hand_num = 3
        if joints_gt is None:
            for j in range(21*hand_num):
                mpix[j] = np.mean(np.stack(mpix[j]))
                # joint_name = self.skeleton[j]['name']
                print('lms_{0}:{1}'.format(j,mpix[j])) 
            # print(eval_summary)
            print('MPJPE_lms: %.2f' % (np.mean(mpix[:63])))
            with open(score_path, 'a') as fo:
                fo.write('UV_mean2d: %f\n' % np.mean(mpix[:63]))
            print('Scores written to: %s' % score_path)

            return None, None

        for j in range(21*hand_num):
            mpjpe[j] = np.mean(np.stack(mpjpe[j]))
            mpix[j] = np.mean(np.stack(mpix[j]))
            # joint_name = self.skeleton[j]['name']
            print('joint_{0}:{1}'.format(j,mpjpe[j])) 
            print('lms_{0}:{1}'.format(j,mpix[j])) 
        # print(eval_summary)
        print('MPJPE_joint: %.2f' % (np.mean(mpjpe[:42])))
        print('MPJPE_lms: %.2f' % (np.mean(mpix[:42])))
        with open(score_path, 'a') as fo:
            fo.write('UV_mean2d: %f\n' % np.mean(mpix[:42]))
            fo.write('UV_mean3d: %f\n' % np.mean(mpjpe[:42]))
        print('Scores written to: %s' % score_path)

        return None, None