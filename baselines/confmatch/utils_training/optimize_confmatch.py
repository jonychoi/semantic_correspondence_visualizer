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

sys.path.append('.')
# from data.dataset import CorrespondenceDataset, TpsGridGen
from utils_training.gen_transform import transform_by_grid

from confmatch.confmatch import ConfMatch
from confmatch.confmatch_loss import EPE
from confmatch.confmatch_loss import contrastive_loss
from confmatch.utils import flow2kps
from confmatch.evaluation import Evaluator
from confmatch.vis import unnormalise_and_convert_mapping_to_flow
from utils_training.utils_confidence import get_pckthres, confidence_loss, estimate_auc, flow2kps_con, calculate_confusion 
from sklearn.metrics import *
from utils_training.gen_transform import cutout_aware
from utils_training.visualize_matching import visualize_matching
from utils_training.visualize_confidence import visualize_actual, visualize_actual_self, visualize

def calculate_mask(sparse_mask, confidence_gt_map, confidence_map, cut_off):
    mask_actual_true = (sparse_mask == True) & (confidence_gt_map == True)
    mask_actual_false = (sparse_mask == True) & (confidence_gt_map == False)

    mask_pred_true = (sparse_mask == True) & (confidence_map.squeeze(1).ge(cut_off) == True)
    mask_pred_false = (sparse_mask == True) & (confidence_map.squeeze(1).ge(cut_off) == False)

    mask_TP = (mask_actual_true == True) & (mask_pred_true == True)
    mask_TN = (mask_actual_false == True) & (mask_pred_false == True)
    mask_FN = (mask_actual_true == True) & (mask_pred_false == True)
    mask_FP = (mask_actual_false == True) & (mask_pred_true == True)
    return ((mask_actual_true, mask_actual_false),
            (mask_pred_true, mask_pred_false),
            (mask_TP, mask_TN, mask_FN, mask_FP))
def convert_unNormFlow_to_unNormMap(flow):
    B, _, H, W = flow.size()
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    map = flow + grid.cuda()
    return map       
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

def cal_strong_src_bbox(gt_bbox, index2D, bbox_size):
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

    x1 = x1+2
    x2 = x2-2
    y1 = y1+2
    y2 = y2-2

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

def cal_strong_trg_bbox(gt_bbox, bbox_size):
    B, H, W = bbox_size
    x1, y1, x2, y2 = torch.round(gt_bbox[:,0] / W).long(),\
                        torch.round(gt_bbox[:,1] / H).long(),\
                        torch.round(gt_bbox[:,2] / W).long(),\
                        torch.round(gt_bbox[:,3] / H).long()
    x1 = x1+2
    x2 = x2-2
    y1 = y1+2
    y2 = y2-2

    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W).repeat(B, 1, 1).cuda()
    yy = yy.view(1, H, W).repeat(B, 1, 1).cuda()
    trg_bbox_mask = (xx >= x1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (xx <= x2.repeat_interleave(H*W).view(-1, H, W)) & \
                    (yy >= y1.repeat_interleave(H*W).view(-1, H, W)) & \
                    (yy <= y2.repeat_interleave(H*W).view(-1, H, W))
    return trg_bbox_mask



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
                optimizer,
                optimizer_con,
                train_loader,
                device,
                epoch,
                train_writer,
                args,
                save_path):
    n_iter = epoch*len(train_loader)

    net.train()
    conf_net.train()
    running_total_loss = 0

    loss_file = '{}_loss_file.txt'.format(args.time_stamp)
    criterion = nn.CrossEntropyLoss()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    assert args.strong_sup_loss == False
    assert args.additional_weak == True
    for i, mini_batch in pbar:
        optimizer.zero_grad()
        optimizer_con.zero_grad()
        #weak
        if args.strong_sup_loss and not args.additional_weak:
            pred_map_weak, corr_weak, occ_S_Tvec, occ_T_Svec =\
                net(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)
            raise NotImplementedError                    
        elif not args.strong_sup_loss and not args.additional_weak:
            pred_flow_weak, pred_map_weak, corr_weak, confidence_map  =\
                confmatch(mini_batch['trg_additional_weak'].to(device),
                        mini_batch['src_img'].to(device), 
                        branch = 'conf',
                        epoch=epoch, n_iter=n_iter, it=i)
            pred_flow_gt_sup = pred_flow_weak
            raise NotImplementedError                     
        # elif not args.strong_sup_loss and args.additional_weak:
        #     pred_map_gt_sup, _, _, _ =\
        #         net(mini_batch['trg_img_weak'].to(device),
        #             mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)    # GT에 쓸거
        #     pred_map_weak_unsup, corr_weak, occ_S_Tvec, occ_T_Svec =\
        #         net(mini_batch['trg_additional_weak'].to(device),
        #             mini_batch['src_img'].to(device), vis_fbcheck=args.use_fbcheck_mask)    # Unsup에 쓸거
        #     pred_map_weak = (pred_map_weak_gt, pred_map_weak_unsup)
        elif not args.strong_sup_loss and args.additional_weak:
            pred_flow_gt_sup, pred_map_gt_sup, _, _ =\
                confmatch(mini_batch['trg_img_weak'].to(device),
                        mini_batch['src_img'].to(device), 
                        epoch=epoch, n_iter=n_iter, it=i)    # GT에 쓸거             
            pred_flow_weak, pred_map_weak, corr_weak, confidence_map  =\
                confmatch(mini_batch['trg_additional_weak'].to(device),
                        mini_batch['src_img'].to(device), 
                        branch = 'conf',
                        epoch=epoch, n_iter=n_iter, it=i)
        else:
            raise NotImplementedError
        
        #strong#
        # if args.aug_mixup != 0 :
        #     ratio = random.random() * args.aug_mixup
        #     mini_batch['trg_img_strong'] = mini_batch['trg_img_strong'] * (1-ratio) + torch.flip(mini_batch['trg_img_strong'],[0]) * ratio
        
        # if args.keymix:
        #     mini_batch['src_img'], mini_batch['trg_img_strong'] =\
        #         confmatch.keypoint_cutmix(image_s=mini_batch['src_img'], kpoint_s=mini_batch['src_kps'], 
        #             image_t=mini_batch['trg_img_strong'], kpoint_t=mini_batch['trg_kps'], 
        #             mask_size_min=5, mask_size_max=20, p=args.keymix, batch_size=mini_batch['trg_img'].size(0),
        #             n_pts=mini_batch['n_pts'])
        if args.keyout:
            mini_batch['trg_img_strong'] = cutout_aware(
                image= mini_batch['trg_img_strong'].to(device), 
                kpoint=mini_batch['trg_kps'], p=args.keyout, cut_n=10, 
                batch_size=mini_batch['trg_img'].size(0), bbox_trg=mini_batch['trg_bbox'],
                n_pts=mini_batch['n_pts'], cutout_size_min=args.keyout_size[0], cutout_size_max=args.keyout_size[1])
        
        mini_batch['trg_img_strong'] = transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device), mode=args.aug_mode) 
        pred_flow_strong, pred_map_strong, corr_strong, _ =\
                confmatch(mini_batch['trg_img_strong'].to(device),
                    mini_batch['src_img'].to(device), 
                    )
        #matching_loss#
        B, corrdim, W, H = corr_weak.size(0), args.feature_size * args.feature_size, args.feature_size, args.feature_size
        B, _, img_H , img_W = mini_batch['src_img'].size()
        B, _, feat_H , feat_W = mini_batch['flow'].size()
        #sup
        flow_gt = mini_batch['flow'].to(device)
        if args.additional_weak:
            loss_sup = EPE(pred_flow_gt_sup, flow_gt, weight=None)
        #unsup-generating_masking
        if args.matching_unsup_position == 'argmax':
            #weak (for vis)
            corr_weak = corr_weak.view(B, corrdim, H, W)
            if args.data_parallel:
                corr_weak_prob = confmatch.module.softmax_with_temperature(corr_weak, args.semi_softmax_corr_temp)            
            else:
                corr_weak_prob = confmatch.softmax_with_temperature(corr_weak, args.semi_softmax_corr_temp)            

            _, index_weak_T_Svec = torch.max(corr_weak_prob, dim=1)            
            src_bbox_mask = cal_strong_src_bbox(mini_batch['src_bbox'].cuda(), index_weak_T_Svec, (B, feat_H, feat_W))
            trg_bbox_mask = cal_strong_trg_bbox(mini_batch['trg_bbox'].cuda(), (B, feat_H, feat_W))


            #weak_transformed
            corr_weak_transformed = transform_by_grid(corr_weak, mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                            interpolation_mode=args.interpolation_mode)
            if args.data_parallel:
                corr_weak_transformed_prob = confmatch.module.softmax_with_temperature(corr_weak_transformed, args.semi_softmax_corr_temp)
            else:
                corr_weak_transformed_prob = confmatch.softmax_with_temperature(corr_weak_transformed, args.semi_softmax_corr_temp)              
            score_weak_T_Svec_transformed, index_weak_T_Svec_transformed = torch.max(corr_weak_transformed_prob, dim=1)
            if args.use_maxProb:
                score_mask_transformed = score_weak_T_Svec_transformed.ge(args.p_cutoff)
            else:
                confidence_map_transformed = transform_by_grid(confidence_map, mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                                interpolation_mode=args.interpolation_mode).squeeze(1)              
                score_mask_transformed = confidence_map_transformed.ge(args.p_cutoff)

            #cal_mask
            assert args.contrastive_gt_mask == True
            if args.contrastive_gt_mask:
                mask2D =  ~(flow_gt[:,0,:,:].bool()) & ~(flow_gt[:,1,:,:].bool())
                mask2D_gt = ~mask2D
                mask2D_gt_transformed = transform_by_grid(mask2D_gt.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                                interpolation_mode=args.interpolation_mode).squeeze().bool()            
            src_bbox_mask_transformed = cal_strong_src_bbox(mini_batch['src_bbox'].cuda(), index_weak_T_Svec_transformed, (B, feat_H, feat_W))
            trg_bbox_mask_transformed = transform_by_grid(trg_bbox_mask.float().unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode,
                                                interpolation_mode=args.interpolation_mode).squeeze().bool()
            mask2D_transformed = mask2D_gt_transformed & score_mask_transformed & src_bbox_mask_transformed & trg_bbox_mask_transformed 
        else:
            raise NotImplementedError                    


        if args.matching_unsup_loss_type == 'contrastive':
            if args.data_parallel:
                loss = confmatch.module.criterion
            else:
                loss = confmatch.criterion            
            corr_strong = corr_strong.view(B, corrdim, H, W)
            loss_unsup = contrastive_loss(loss, \
                                                corr_strong, index_weak_T_Svec_transformed, mask2D_transformed, \
                                                device = device,
                                                use_loss_weight=args.use_confidence_weight,
                                                contrastive_temp = args.semi_contrastive_temp)
        else:
            raise NotImplementedError             
        #to_do: loss_EPE
        #confidence_loss#
        if args.data_parallel:
            loss_con = confmatch.module.loss_con
        else:
            loss_con = confmatch.loss_con
        down_ratio = feat_H / img_H        
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

        src_bbox_mask_self = cal_src_bbox(mini_batch['trg_bbox'].cuda(), pred_map_self, (B, feat_H, feat_W))
        self_mask = mask2D_self_gt.squeeze(1) & trg_bbox_mask_self & src_bbox_mask_self
        # self_mask = mask2D_self_gt.squeeze(1) & trg_bbox_mask_self 
        margin_thres_self = get_pckthres(mini_batch['trg_bbox'].cuda(), down_ratio, args.alpha_train_self)
        loss_EPE_map, confidence_gt_map_self = calculate_gt_confidence_map_by_flow(pred_flow_self, gt_flow_self, margin_thres_self)
        loss_con.reduction = 'none'
        confidence_map_self = confidence_map_self.squeeze(1)
        mask2D_self_gt = mask2D_self_gt.squeeze(1)
        loss_con_map_self = loss_con(confidence_map_self, confidence_gt_map_self.clone().detach().float())
        loss_con_self = torch.mean(loss_con_map_self[self_mask])
        Loss_con = loss_con_sup + loss_con_self

        Loss_con.backward()
        optimizer_con.step()
                
        # Loss = loss_sup + loss_unsup
        if args.dynamic_unsup :
            if loss_unsup == 0:
                loss_unsup = torch.tensor(0.0, device=device)
            else:
                loss_unsup = (loss_sup.detach() / loss_unsup.detach()) * loss_unsup
            Loss = loss_sup + loss_unsup
        if args.zero_sup :
            loss_sup = torch.tensor(0.0, device=device)
            Loss = loss_sup + loss_unsup

        Loss.backward()
        optimizer.step()

        running_total_loss += Loss.item() 

        with open(os.path.join(args.cur_snapshot, loss_file), 'a+') as file:
            # file.write(f'loss:{loss_sup.item(), loss_unsup.item()}|diff_ratio:{diff_ratio.item()}\n')
            file.write(f'loss:{loss_sup.item(), loss_unsup.item()}, loss_con:{loss_con_sup.item(), loss_con_self.item()} \n')
        train_writer.add_scalar('train_loss_per_iter', Loss.item(), n_iter)
        train_writer.add_scalar('train_loss_con_per_iter', Loss_con.item(), n_iter)

        #visualization#
        '''
        matching_vis
        '''
        if n_iter % args.vis_interval == 0:        
            if mask2D[0].sum() == 0:
                pass
            else:
                #weak (for vis)
                corr_strong = corr_strong.view(B, corrdim, H, W)
                if args.data_parallel:
                    corr_strong_prob = confmatch.module.softmax_with_temperature(corr_strong, args.semi_softmax_corr_temp)            
                else:
                    corr_strong_prob = confmatch.softmax_with_temperature(corr_strong, args.semi_softmax_corr_temp)                     
                _, index_strong = torch.max(corr_strong_prob, dim=1)            
                
                # Visualization Variable(weak_not_T)#
                mask_tgt_kp2D_weak = (mask2D[0] == True).nonzero(as_tuple=False).transpose(-1,-2)
                #type1 (by_softArgmax)
                # mask_tgt_kp2D_weak_not_T = pred_map_weak_S_Tvec[0].permute(1,2,0)[mask2D_not_T[0] == True]
                # mask_tgt_kp2D_weak_not_T = mask_tgt_kp2D_weak_not_T.transpose(-1,-2)
                # print("mask_src_kp2D_weak_not_T", mask_src_kp2D_weak_not_T)
                # print("mask_tgt_kp2D_weak_not_T", mask_tgt_kp2D_weak_not_T)
                #type2 (by_Argmax)
                mask_src_1D = index_weak_T_Svec[0][mask2D[0]]
                mask_src_kp2D_weak = torch.cat(((mask_src_1D // W).view(1,-1), (mask_src_1D % W).view(1,-1)), dim=0)
                # Visualization Variable(weak_T)#        
                # mask_src_kp2D_weak = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_tgt_kp2D_weak_transformed = (mask2D_transformed[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D_transformed = index_weak_T_Svec_transformed[0][mask2D_transformed[0]]
                mask_src_kp2D_weak_transformed = torch.cat(((mask_src_1D_transformed // W).view(1,-1), (mask_src_1D_transformed % W).view(1,-1)), dim=0)
                
                # Visualization Variable(strong_T)        
                mask_tgt_kp2D_strong = (mask2D_transformed[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
                mask_src_1D = index_strong[0][mask2D_transformed[0]]
                mask_src_kp2D_strong = torch.cat(((mask_src_1D //  W).view(1,-1), (mask_src_1D % W).view(1,-1)), dim=0)
                weak_transformed = transform_by_grid(
                                    mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),
                                    mini_batch[args.aug_mode][0].unsqueeze(0).to(device), 
                                    mode=args.aug_mode)
                diff_point = 0.0
                diff_idx = []
                for idx, (weak_pt, strong_pt) in enumerate(zip(mask_src_kp2D_weak_transformed.permute(-1,-2), mask_src_kp2D_strong.permute(-1,-2))):  # weak_pt strong_pt (x,y)
                    dist = torch.sqrt((weak_pt[0] - strong_pt[0]).pow(2) + (weak_pt[1] - strong_pt[1]).pow(2))
                    if dist >= 1:
                        diff_point += 1.0
                        diff_idx.append(idx)
                diff_idx = torch.tensor(diff_idx)
                if diff_point != 0:
                    diff_point /= mask_tgt_kp2D_strong.size(1)
                visualize_matching(mini_batch,
                            mask_src_kp2D_weak,
                            mask_tgt_kp2D_weak,
                            mask_src_kp2D_weak_transformed,
                            mask_tgt_kp2D_weak_transformed,
                            mask_src_kp2D_strong,
                            mask_tgt_kp2D_strong,
                            weak_transformed,
                          device, args, n_iter, diff_idx)

            '''
            conf_vis - sup
            '''        
            mask_all = calculate_mask(mask2D_gt, confidence_gt_map, confidence_map, args.cut_off)
            mask_actual_true, mask_actual_false = mask_all[0]
            mask_pred_true, mask_pred_false = mask_all[1]
            mask_TP, mask_TN, mask_FN, mask_FP = mask_all[2]                    

            gt_map = convert_unNormFlow_to_unNormMap(flow_gt)
            ori_tgt_kp2D = (mask2D_gt[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            ori_src_kp2D = torch.cat((gt_map[0][1][mask2D_gt[0]].view(1, -1), gt_map[0][0][mask2D_gt[0]].view(1, -1)), dim=0) 

            actual_true_tgt_kp2D = (mask_actual_true[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            actual_true_src_kp2D = torch.cat((pred_map_weak[0][1][mask_actual_true[0]].view(1, -1), pred_map_weak[0][0][mask_actual_true[0]].view(1, -1)), dim=0) 

            actual_false_tgt_kp2D = (mask_actual_false[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            actual_false_src_kp2D = torch.cat((pred_map_weak[0][1][mask_actual_false[0]].view(1, -1), pred_map_weak[0][0][mask_actual_false[0]].view(1, -1)), dim=0) 

            pred_true_tgt_kp2D = (mask_pred_true[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            pred_true_src_kp2D = torch.cat((pred_map_weak[0][1][mask_pred_true[0]].view(1, -1), pred_map_weak[0][0][mask_pred_true[0]].view(1, -1)), dim=0) 

            pred_false_tgt_kp2D = (mask_pred_false[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            pred_false_src_kp2D = torch.cat((pred_map_weak[0][1][mask_pred_false[0]].view(1, -1), pred_map_weak[0][0][mask_pred_false[0]].view(1, -1)), dim=0) 

            TP_tgt_kp2D = (mask_TP[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            TP_src_kp2D = torch.cat((pred_map_weak[0][1][mask_TP[0]].view(1, -1), pred_map_weak[0][0][mask_TP[0]].view(1, -1)), dim=0) 

            TN_tgt_kp2D = (mask_TN[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            TN_src_kp2D = torch.cat((pred_map_weak[0][1][mask_TN[0]].view(1, -1), pred_map_weak[0][0][mask_TN[0]].view(1, -1)), dim=0) 

            FN_tgt_kp2D = (mask_FN[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            FN_src_kp2D = torch.cat((pred_map_weak[0][1][mask_FN[0]].view(1, -1), pred_map_weak[0][0][mask_FN[0]].view(1, -1)), dim=0) 

            FP_tgt_kp2D = (mask_FP[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            FP_src_kp2D = torch.cat((pred_map_weak[0][1][mask_FP[0]].view(1, -1), pred_map_weak[0][0][mask_FP[0]].view(1, -1)), dim=0) 
            visualize(mini_batch,
                    ori_src_kp2D, ori_tgt_kp2D,
                    actual_true_src_kp2D, actual_true_tgt_kp2D,
                    actual_false_src_kp2D, actual_false_tgt_kp2D,
                    pred_true_src_kp2D, pred_true_tgt_kp2D,
                    pred_false_src_kp2D, pred_false_tgt_kp2D,
                    TP_src_kp2D, TP_tgt_kp2D,
                    TN_src_kp2D, TN_tgt_kp2D,
                    FN_src_kp2D, FN_tgt_kp2D,
                    FP_src_kp2D, FP_tgt_kp2D,
                    device, args, n_iter)
            '''
            conf_vis - self
            '''
            #vis (y,x) coordinate
            gt_map_self = gt_map_self.clamp(0, 15)
            mask_actual_true = (self_mask == True) & (confidence_gt_map_self == True)
            mask_actual_false = (self_mask == True) & (confidence_gt_map_self == False)
            actual_true_tgt_kp2D = (mask_actual_true[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            actual_true_src_kp2D = torch.cat((gt_map_self[0][1][mask_actual_true[0]].view(1, -1), 
                                                gt_map_self[0][0][mask_actual_true[0]].view(1, -1)), dim=0)                                     
            actual_false_tgt_kp2D = (mask_actual_false[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            actual_false_src_kp2D = torch.cat((gt_map_self[0][1][mask_actual_false[0]].view(1, -1), 
                                                gt_map_self[0][0][mask_actual_false[0]].view(1, -1)), dim=0) 
            # visualize_actual_self(mini_batch,
            #         actual_true_src_kp2D.int(), actual_true_tgt_kp2D.int(),
            #         actual_false_src_kp2D.int(), actual_false_tgt_kp2D.int(),
            #         device, args, n_iter)
            visualize_actual(mini_batch,
                    actual_true_src_kp2D, actual_true_tgt_kp2D,
                    actual_false_src_kp2D, actual_false_tgt_kp2D,
                    device, args, n_iter, plot_name='actual_self_GT_{}'.format(n_iter))

            pred_true_tgt_kp2D = (mask_actual_true[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            pred_true_src_kp2D = torch.cat((pred_map_self[0][1][mask_actual_true[0]].view(1, -1), 
                                                pred_map_self[0][0][mask_actual_true[0]].view(1, -1)), dim=0)                                     
            pred_false_tgt_kp2D = (mask_actual_false[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
            pred_false_src_kp2D = torch.cat((pred_map_self[0][1][mask_actual_false[0]].view(1, -1), 
                                                pred_map_self[0][0][mask_actual_false[0]].view(1, -1)), dim=0)    
            visualize_actual(mini_batch,
                    pred_true_src_kp2D, pred_true_tgt_kp2D,
                    pred_false_src_kp2D, pred_false_tgt_kp2D,
                    device, args, n_iter, plot_name='actual_self_pred_{}'.format(n_iter))     

        # train_writer.add_scalar('diff_point', diff_ratio, n_iter)
        n_iter += 1
        # pbar.set_description(
        #     f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_sup.item():.3f}|UnsupLoss:{loss_unsup.item():.3f}|SelfsupLoss:{loss_self.item():.3f}|diff_ratio:{diff_ratio.item():.3f}')
        pbar.set_description(
            f'training: R_total_loss:{(running_total_loss / (i + 1)):.3f}/{Loss.item():.3f}|SupLoss:{loss_sup.item():.3f}|UnsupLoss:{loss_unsup.item():.3f}|Con_supLoss:{loss_con_sup.item():.3f}|Con_selfLoss"{loss_con_self.item():.3f}')
    running_total_loss /= len(train_loader)
    return running_total_loss


def validate_epoch(confmatch,
                   val_loader,
                   device,
                   epoch):
    confmatch.eval()
    running_total_loss = 0

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            flow_gt = mini_batch['flow'].to(device)
            pred_flow, pred_map, corr , confidence_map = confmatch(mini_batch['trg_img'].to(device),
                                    mini_batch['src_img'].to(device))
            # pred_flow = unnormalise_and_convert_mapping_to_flow(pred_map)

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