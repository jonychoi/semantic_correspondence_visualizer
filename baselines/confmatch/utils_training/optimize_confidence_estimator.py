import pdb
import time
import sys
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append('.')
# from data.dataset import CorrespondenceDataset, TpsGridGen
from utils_training.gen_transform import transform_by_grid

from confmatch.confmatch import ConfMatch
from confmatch.confmatch_loss import EPE
from confmatch.confmatch_loss import contrastive_loss
from confmatch.utils import flow2kps
from confmatch.evaluation import Evaluator
from confmatch.vis import unnormalise_and_convert_mapping_to_flow
from utils_training.utils_confidence import confidence_loss, estimate_auc, flow2kps_con, calculate_confusion

def cal_src_bbox(gt_bbox, index2D, bbox_size):
    B, H, W = bbox_size
    if index2D.size(1) == 2:
        x_T_Svec = index2D[:,0]
        y_T_Svec = index2D[:,1]
    else:
        x_T_Svec, y_T_Svec = (index2D % W), (index2D // W)
    x1, y1, x2, y2 = torch.round(gt_bbox[:,0] / W).long(),\
                        torch.round(gt_bbox[:,1] / H).long(),\
                        torch.round(gt_bbox[:,2] / W).long(),\
                        torch.round(gt_bbox[:,3] / H).long()
    #src_mask#
    src_bbox_mask = (x_T_Svec >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (x_T_Svec <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                    (y_T_Svec >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (y_T_Svec <= y2.repeat_interleave(H*W).view(-1, H, W))
    return src_bbox_mask

def cal_trg_bbox(gt_bbox, bbox_size):
    B, H, W = bbox_size
    x1, y1, x2, y2 = torch.round(gt_bbox[:,0] / W).long(),\
                        torch.round(gt_bbox[:,1] / H).long(),\
                        torch.round(gt_bbox[:,2] / W).long(),\
                        torch.round(gt_bbox[:,3] / H).long()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W).repeat(B, 1, 1).cuda()
    yy = yy.view(1, H, W).repeat(B, 1, 1).cuda()
    trg_bbox_mask = (xx >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (xx <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                    (yy >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (yy <= y2.repeat_interleave(H*W).view(-1, H, W))
    return trg_bbox_mask

def get_pckthres(bbox, down_ratio, alpha):
    r"""Computes PCK threshold"""
    bbox_w = (bbox[:,2] - bbox[:,0])
    bbox_h = (bbox[:,3] - bbox[:,1])
    pckthres = torch.max(bbox_w, bbox_h)
    return torch.tensor(pckthres.float() * down_ratio * alpha).float()

def calculate_gt_confidence_map_by_flow(input_flow, target_flow, margin_thres):
    B,_,h,w = target_flow.size()
    EPE_map = torch.norm(target_flow-input_flow, 2, 1) # EPE_map.size() :torch.Size([32, 16, 16])
    margin_thres = margin_thres.unsqueeze(1).expand(B, h*w).view(B, h, w)
    confidence_gt_map = EPE_map.le(margin_thres)
    return EPE_map, confidence_gt_map.squeeze(1)

def gen_Norm_identity_map(B, H, W):
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).cuda().float()
    grid[:,0,:,:] = 2 * grid[:,0,:,:].clone() / (W-1) -1
    grid[:, 1, :, :] = 2 * grid[:, 1, :, :].clone() / (H - 1) - 1    
    return grid  


    

def train_epoch(confmatch:ConfMatch,
                net,
                conf_net,
                optimizer_con,
                train_loader,
                device,
                epoch,
                train_writer,
                args,
                save_path):
    n_iter = epoch*len(train_loader)

    net.eval()
    conf_net.train()
    running_total_loss = 0

    loss_file = '{}_loss_file.txt'.format(args.time_stamp)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    assert args.strong_sup_loss == False
    assert args.additional_weak == True
    for i, mini_batch in pbar:
        optimizer_con.zero_grad()
        #weak
        if args.strong_sup_loss and not args.additional_weak:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
            raise NotImplementedError                    
        elif not args.strong_sup_loss and not args.additional_weak:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_weak'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
            raise NotImplementedError                     

        elif not args.strong_sup_loss and args.additional_weak:
           # GT에 쓸거             
            pred_flow_weak, pred_map_weak, corr_weak, confidence_map  =\
                confmatch(mini_batch['trg_additional_weak'].to(device),
                        mini_batch['src_img'].to(device), 
                        branch = 'conf',
                        epoch=epoch, n_iter=n_iter, it=i)
        else:
            raise NotImplementedError
        

        
        mini_batch['trg_img_strong'] = transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device), mode=args.aug_mode) 
         
        #to_do: loss_EPE
        #confidence_loss#
        loss_con = confmatch.loss_con
        B, _, img_H , img_W = mini_batch['src_img'].size()
        B, _, feat_H , feat_W = mini_batch['flow'].size()        
        down_ratio = feat_H / img_H
        flow_gt = mini_batch['flow'].to(device)
        mask2D =  ~(flow_gt[:,0,:,:].bool()) & ~(flow_gt[:,1,:,:].bool())
        mask2D_gt = ~mask2D
        trg_bbox_mask = cal_trg_bbox(mini_batch['trg_bbox'].cuda(), (B, feat_H, feat_W))        
        margin_thres = get_pckthres(mini_batch['src_bbox'].cuda(), down_ratio, args.alpha_train)
        loss_EPE_map, confidence_gt_map = calculate_gt_confidence_map_by_flow(pred_flow_weak, flow_gt, margin_thres)
        loss_con.reduction = 'none'
        confidence_map = confidence_map.squeeze(1)
        mask2D_gt = mask2D_gt.squeeze(1)
        loss_con_map = loss_con(confidence_map,confidence_gt_map.clone().detach().float())
        loss_con_sup = torch.mean(loss_con_map[mask2D_gt])

        pred_flow_self, pred_map_self, _, confidence_map_self  = confmatch(mini_batch['trg_img_strong'].to(device),
                                        mini_batch['trg_img_weak'].to(device), 
                                        branch='conf',
                                        epoch=epoch, n_iter=n_iter, it=i)
            
        mask2D_self_gt = torch.ones_like(confidence_map_self)
        mask2D_self_gt = transform_by_grid(mask2D_self_gt, mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                        interpolation_mode=args.interpolation_mode).squeeze(1)
        mask2D_self_gt = mask2D_self_gt.bool()
        norm_identity_map = gen_Norm_identity_map(B, feat_H, feat_W)
        gt_nom_map_self = transform_by_grid(norm_identity_map, mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                        interpolation_mode=args.interpolation_mode)
        gt_flow_self, gt_map_self = unnormalise_and_convert_mapping_to_flow(gt_nom_map_self, use_map=True)
        #src(trg) - trg(transformed)
        trg_bbox_mask_self = transform_by_grid(trg_bbox_mask.float().unsqueeze(1), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                        interpolation_mode=args.interpolation_mode).bool().squeeze(1) 

        # src_bbox_mask_self = cal_src_bbox(mini_batch['trg_bbox'].cuda(), pred_map_self, (B, feat_H, feat_W))
        # self_mask = mask2D_self_gt.squeeze(1) & trg_bbox_mask_self & src_bbox_mask_self
        self_mask = mask2D_self_gt.squeeze(1) & trg_bbox_mask_self 
        margin_thres_self = get_pckthres(mini_batch['trg_bbox'].cuda(), down_ratio, args.alpha_train_self)
        loss_EPE_map, confidence_gt_map = calculate_gt_confidence_map_by_flow(pred_flow_self, gt_flow_self, margin_thres_self)
        loss_con.reduction = 'none'
        confidence_map_self = confidence_map_self.squeeze(1)
        mask2D_self_gt = mask2D_self_gt.squeeze(1)
        loss_con_map_self = loss_con(confidence_map_self, confidence_gt_map.clone().detach().float())
        loss_con_self = torch.mean(loss_con_map_self[self_mask])
        Loss = loss_con_sup + loss_con_self

        Loss.backward()
        optimizer_con.step()

        running_total_loss += Loss.item()

        with open(os.path.join(args.cur_snapshot, loss_file), 'a+') as file:
            # file.write(f'loss:{loss_sup.item(), loss_unsup.item()}|diff_ratio:{diff_ratio.item()}\n')
            file.write(f'loss:{loss_con_sup.item(), loss_con_self.item()} \n')
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        # train_writer.add_scalar('diff_point', diff_ratio, n_iter)
        n_iter += 1
        # pbar.set_description(
        #     f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_sup.item():.3f}|UnsupLoss:{loss_unsup.item():.3f}|SelfsupLoss:{loss_self.item():.3f}|diff_ratio:{diff_ratio.item():.3f}')
        pbar.set_description(
            f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_con_sup.item():.3f}|UnsupLoss:{loss_con_self.item():.3f}')
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(net,
                   val_loader,
                   device,
                   epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            # mask2D =  ~(flow_gt[:,0,:,:].bool()) & ~(flow_gt[:,1,:,:].bool())
            # mask2D_gt = ~mask2D            
            # pdb.set_trace()
            pred_map, _, _, _ = net(mini_batch['trg_img'].to(device),
                                    mini_batch['src_img'].to(device))
            pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result['pck']

            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

    return running_total_loss / len(val_loader), mean_pck


def validate_epoch_test(net,val_loader,device,epoch):
    net.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        pck_cat_array = dict()
        mean_pck_cat_array = dict()
        for i, mini_batch in pbar:
            
            flow_gt = mini_batch['flow'].to(device)
            pred_map, _, _, _ = net(mini_batch['trg_img'].to(device),
                                    mini_batch['src_img'].to(device))
            pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)

            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))

            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)

            Loss = EPE(pred_flow, flow_gt)

            pck_array += eval_result['pck']

            for idx, cat in enumerate(mini_batch['category']) :
                if cat not in pck_cat_array :
                    pck_cat_array[cat] = []
                pck_cat_array[cat].append(eval_result['pck'][idx])
                
            running_total_loss += Loss.item()
            pbar.set_description(
                ' validation R_total_loss: %.3f/%.3f' % (running_total_loss / (i + 1), Loss.item()))
        mean_pck = sum(pck_array) / len(pck_array)

        for key in pck_cat_array :
            mean_pck_cat_array[key] = sum(pck_cat_array[key]) / len(pck_cat_array[key])



    return running_total_loss / len(val_loader), mean_pck, mean_pck_cat_array

