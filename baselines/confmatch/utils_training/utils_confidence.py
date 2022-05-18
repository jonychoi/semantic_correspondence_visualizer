import pdb
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('..')
from confmatch.utils import flow2kps
from confmatch.evaluation import Evaluator
from confmatch.confmatch_loss import EPE
from confmatch.vis import unnormalise_and_convert_mapping_to_flow
from sklearn.metrics import *

def get_pckthres(bbox, down_ratio, alpha):
    r"""Computes PCK threshold"""
    bbox_w = (bbox[:,2] - bbox[:,0])
    bbox_h = (bbox[:,3] - bbox[:,1])
    pckthres = torch.max(bbox_w, bbox_h)
    return torch.tensor(pckthres.float() * down_ratio * alpha).float()
def validate_epoch_confidence(net,
                   val_loader,
                   device, args,cut_off,alpha,
                   epoch,save_path):
    net.eval()
    running_total_loss = 0
    
    running_max_prob_mean = 0
    running_max_prob_max = 0
    running_max_prob_min = 0

    running_confidence_mean = 0
    running_confidence_max = 0
    running_confidence_min = 0
    
    
    
    confidence_mean_list = []
    confidence_max_list = []
    confidence_min_list = []

    max_prob_mean_list = []
    max_prob_max_list = []
    max_prob_min_list = []

    confusion_matrix_con = {'TP':0,'FP':0,'FN':0,'TN':0}
    confusion_matrix_max_prob = {'TP':0,'FP':0,'FN':0,'TN':0}
    
    label_list_all_flow = []
    label_list_all_flow = np.array(label_list_all_flow)
    confience_list_all_flow = []
    confience_list_all_flow = np.array(confience_list_all_flow)
    max_prob_list_all_flow = []
    max_prob_list_all_flow = np.array(max_prob_list_all_flow)
    loss = nn.BCELoss()

    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        pck_array = []
        for i, mini_batch in pbar:
            
            flow_gt = mini_batch['flow'].to(device)
            pred_flow, corr_map, corr_tensor, confidence_map = net(mini_batch['trg_img'].to(device), # torch.Size([32, 2, 16, 16])
                            mini_batch['src_img'].to(device),branch='conf')
            margin_thres = get_pckthres(mini_batch['src_bbox'].cuda(), 1/16, alpha)

            corr_tensor = corr_tensor.view(-1, 256, 16, 16)
            max_probs_val, _ = torch.max(corr_tensor, dim=1)
        
            confidence_map_list_flow,max_prob_list_flow, label_list_flow= estimate_auc_flow(mini_batch,mini_batch['trg_kps'].to(device),
            mini_batch['src_kps'].to(device),mini_batch['flow'].to(device),mini_batch['src_img'].to(device), pred_flow,corr_map,
            confidence_map,max_probs_val, mini_batch['n_pts'].to(device), margin_thres.to(device))


            
            label_list_all_flow = np.append(label_list_all_flow, label_list_flow)
            confience_list_all_flow =np.append(confience_list_all_flow, confidence_map_list_flow)
            max_prob_list_all_flow = np.append(max_prob_list_all_flow, max_prob_list_flow)

            confusion_mask_num = calculate_confusion(torch.tensor(label_list_all_flow).bool(), torch.tensor(confience_list_all_flow), cut_off)
            mask_TP, mask_TN, mask_FN, mask_FP = confusion_mask_num
            
            confusion_matrix_con['TP'] = confusion_matrix_con['TP'] + mask_TP.item()
            confusion_matrix_con['FP'] = confusion_matrix_con['FP'] + mask_FP.item()
            confusion_matrix_con['FN'] = confusion_matrix_con['FN'] + mask_FN.item()
            confusion_matrix_con['TN'] = confusion_matrix_con['TN'] + mask_TN.item()
            
            confusion_mask_num = calculate_confusion(torch.tensor(label_list_all_flow).bool(), torch.tensor(max_prob_list_all_flow), cut_off)
            mask_TP, mask_TN, mask_FN, mask_FP = confusion_mask_num         

            confusion_matrix_max_prob['TP'] = confusion_matrix_max_prob['TP'] + mask_TP.item()
            confusion_matrix_max_prob['FP'] = confusion_matrix_max_prob['FP'] + mask_FP.item()
            confusion_matrix_max_prob['FN'] = confusion_matrix_max_prob['FN'] + mask_FN.item()
            confusion_matrix_max_prob['TN'] = confusion_matrix_max_prob['TN'] + mask_TN.item()
            # if args.data_parallel:
            #     pred_flow = unnormalise_and_convert_mapping_to_flow(corr_map)
            # else:
            #     pred_flow = unnormalise_and_convert_mapping_to_flow(corr_map)                
            estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
            eval_result = Evaluator.eval_kps_transfer(estimated_kps.cpu(), mini_batch)
            pck_array += eval_result['pck']
            Loss = EPE(pred_flow, flow_gt)
            running_total_loss += Loss.item()

        avg_confidence_mean = torch.mean(torch.tensor(confience_list_all_flow))
        avg_confidence_max =  torch.max(torch.tensor(confience_list_all_flow))
        avg_confidence_min =  torch.min(torch.tensor(confience_list_all_flow))
        avg_max_prob_mean =  torch.mean(torch.tensor(max_prob_list_all_flow))
        avg_max_prob_max =  torch.max(torch.tensor(max_prob_list_all_flow))
        avg_max_prob_min =  torch.min(torch.tensor(max_prob_list_all_flow))
        scores_dictionary = {}
        scores_dictionary['AUC_con'] = roc_auc_score(label_list_all_flow, confience_list_all_flow)
        scores_dictionary['AUC_max_prob'] = roc_auc_score(label_list_all_flow, max_prob_list_all_flow)
        scores_dictionary['avg_conf_mean'] = avg_confidence_mean.item()
        scores_dictionary['avg_conf_max'] = avg_confidence_max.item()
        scores_dictionary['avg_conf_min'] = avg_confidence_min.item()
        scores_dictionary['avg_max_prob_mean'] = avg_max_prob_mean.item()
        scores_dictionary['avg_max_prob_max'] = avg_max_prob_max.item()
        scores_dictionary['avg_max_prob_min'] = avg_max_prob_min.item()

        

        mean_pck = sum(pck_array) / len(pck_array)
        
    return running_total_loss / len(val_loader), mean_pck ,confusion_matrix_con,confusion_matrix_max_prob,scores_dictionary

def confidence_loss(trg_kps, src_kps_gt, corr_map, confidence_map, max_probs_val, n_pts, margin_thres):
    B, _, h, w = corr_map.size() # h=16, w=16
    
    
    confidence_map_mean = 0.0
    confidence_map_max = 0.0
    confidence_map_min = 0.0

    max_prob_mean = 0.0
    max_prob_max = 0.0
    max_prob_min = 0.0
    estimated_confidence_map_list =[]
    estimated_max_prob_list = []
    max_probs_val = max_probs_val.unsqueeze(1)
    for trg_kps,src_kps_gt, corr_map, confidence_map, n_pts, thres,max_prob in zip(trg_kps.long(),src_kps_gt.long(), corr_map,confidence_map, n_pts,margin_thres,max_probs_val):
        size = trg_kps.size(1) # size: 40, trg_kps.size(): torch.Size([2, 40])

        trg_kps = torch.round(trg_kps / 16)
        src_kps_gt = src_kps_gt / 16
        trg_kps_sparse = (torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, 15)).long() 
        src_kps_sparse = torch.clamp(src_kps_gt.narrow_copy(1, 0, n_pts), 0, 15)

        
        estimated_kps = corr_map[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        
        estimated_confidence_map = confidence_map[:,trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        
        estimated_max_prob = max_prob[:,trg_kps_sparse[1, :], trg_kps_sparse[0, :]]
        # estimated_max_prob = max_prob[kp[1, :], kp[0, :]]
        l2dist = (estimated_kps - src_kps_sparse).pow(2).sum(dim=0).pow(0.5) #torch.Size([11])
        correct_pts = torch.le(l2dist, thres) #tensor([True, True, True, True, True, True, True, True, True, True, True],device='cuda:0') 이런식으로 bool 값 반환
        # pdb.set_trace()
        # max_probs_val_mean = torch.mean(estimated_max_prob)
        estimated_confidence_map_list.extend(estimated_confidence_map.squeeze(0))
        estimated_max_prob_list.extend(estimated_max_prob.squeeze(0))

    
    return estimated_confidence_map_list,estimated_max_prob_list

def estimate_auc(trg_kps,src_kps_gt,corrmap,confidence_map,maxprob_map, n_pts, margin_thres):
    
    estimated_src_kps_list = []

    confidence_map_list = []
    confidence_map_list = np.array(confidence_map_list)

    max_prob_map_list = []
    max_prob_map_list = np.array(max_prob_map_list)

    label_list = []
    label_list = np.array(label_list)
    maxprob_map = maxprob_map.unsqueeze(1)
    for trg_kps,src_kps_gt, corrmap, confidence_map,maxprob_map, n_pts, thres in zip(trg_kps.long(),src_kps_gt.long(), corrmap,confidence_map,maxprob_map, n_pts, margin_thres):
        size = trg_kps.size(1) # size: 40, trg_kps.size(): torch.Size([2, 40])
        # max_prob torch.Size([16, 16])
        trg_kps = torch.round(trg_kps / 16)
        src_kps_gt = src_kps_gt / 16
        trg_kps_sparse = (torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, 15)).long() 
        src_kps_sparse = torch.clamp(src_kps_gt.narrow_copy(1, 0, n_pts), 0, 15)
        
        estimated_src_kps = corrmap[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        sparse_confidence_map = confidence_map[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        sparse_max_prob_map = maxprob_map[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]]
        l2dist = (estimated_src_kps - src_kps_sparse).pow(2).sum(dim=0).pow(0.5) #torch.Size([11])
        thres = thres.expand_as(l2dist).float()         
        correct_pts = torch.le(l2dist, thres)
        max_prob_map_list = np.append(max_prob_map_list,sparse_max_prob_map.squeeze(0).cpu().detach().numpy())
        confidence_map_list = np.append(confidence_map_list,sparse_confidence_map.squeeze(0).cpu().detach().numpy())
        label_list = np.append(label_list, correct_pts.int().cpu().detach().numpy())
        # max_prob_list = np.append(max_prob_list, estimated_max_prob.cpu().detach().numpy())
    
    
    return confidence_map_list,max_prob_map_list,label_list #,max_prob_list

def estimate_auc_flow(mini_batch,trg_kps,src_kps_gt,flow_gt,src_img,flow,corrmap,confidence_map,maxprob_map, n_pts, margin_thres, upsample_size=(256, 256)):
    B, _, img_H , img_W = src_img.size()
    B, _, feat_H , feat_W = flow_gt.size()
    down_ratio = feat_H / img_H
    mask = (flow_gt[:,0] == 0) & (flow_gt[:,1] == 0)
    sparse_mask = ~mask
    EPE_map = torch.norm(flow-flow_gt, 2, 1) # EPE_map.size() :torch.Size([32, 16, 16])
    margin_thres_map = margin_thres.unsqueeze(1).expand(B,feat_H*feat_W).view(B,feat_H,feat_W)
    confidence_map_gt = EPE_map.le(margin_thres_map) # 거리가 마진이하인 점들
    confidence_map_gt_sparse = confidence_map_gt[sparse_mask] # 왜 n_pts != (~mask).count_nonzero() 일까..?
    confidence_map_sparse = confidence_map.squeeze(1)[sparse_mask]
    maxprob_map_sparse = maxprob_map.squeeze(1)[sparse_mask]

    label_list = confidence_map_gt_sparse.cpu().detach().numpy()
    confidence_map_list = confidence_map_sparse.cpu().detach().numpy()
    max_prob_map_list = maxprob_map_sparse.cpu().detach().numpy()
    # pdb.set_trace()
   
    
    return confidence_map_list,max_prob_map_list,label_list 

def flow2kps_con(trg_kps,src_kps_gt, flow,confidence_map,maxprob_map, n_pts, cut_off, margin_thres, upsample_size=(256,256)):
    _, _, h, w = flow.size() # h=16, w=16
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)  #torch.Size([2, 256, 256])
    confidence_map = F.interpolate(confidence_map, upsample_size, mode='bilinear') # 
    maxprob_map = F.interpolate(maxprob_map, upsample_size, mode='bilinear')

   
    pred_dict_con = {'TP':[],'FP':[],'FN':[],'TN':[]}
    pred_dict_number_con = {'TP':[],'FP':[],'FN':[],'TN':[]}
    pred_dict_maxprob = {'TP':[],'FP':[],'FN':[],'TN':[]}
    pred_dict_number_maxprob = {'TP':[],'FP':[],'FN':[],'TN':[]}
   
    for trg_kps,src_kps_gt, flow, confidence_map,maxprob_map, n_pts, thres in zip(trg_kps.long(),src_kps_gt.long(), flow,confidence_map,maxprob_map, n_pts,margin_thres):
        size = trg_kps.size(1) # size: 40, trg_kps.size(): torch.Size([2, 40])
        # max_prob torch.Size([16, 16])
        trg_kps_sparse = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1) # kp.size() :torch.Size([2, 11]), upsample_size: (256, 256) , trg_kps.narrow_copy(1, 0, n_pts) 는 trg_kps 의 0번 인덱스에서 11개 만큼 가져오겠다는 얘기 torch.Size([2, 11]) 임
        src_kps_gt_sparse = torch.clamp(src_kps_gt.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        # 그래서 trg_kps.narrow_copy(1, 0, n_pts)  의 value 를 최소 0 최대 255 로 제한 하겠다는 것.
        estimated_src_kps = trg_kps_sparse + flow[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        estimated_confidence_map = confidence_map[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]] # torch.Size([2, 11])
        estimated_maxprob_map = maxprob_map[:, trg_kps_sparse[1, :], trg_kps_sparse[0, :]]
        
        _,TP_mask_con,FP_mask_con,FN_mask_con,TN_mask_con =  classify_prediction(estimated_src_kps,src_kps_gt_sparse,estimated_confidence_map,thres,cut_off)
        _,TP_mask_maxprob,FP_mask_maxprob,FN_mask_maxprob,TN_mask_maxprob =  classify_prediction(estimated_src_kps,src_kps_gt_sparse,estimated_maxprob_map,thres,cut_off)

      
        
        n_pts = n_pts.item()
        size = size - n_pts
        # pdb.set_trace()

        pred_dict_con['TP'].append(TP_mask_con)
        pred_dict_con['FP'].append(FP_mask_con)
        pred_dict_con['FN'].append(FN_mask_con)
        pred_dict_con['TN'].append(TN_mask_con)

        pred_dict_maxprob['TP'].append(TP_mask_maxprob)
        pred_dict_maxprob['FP'].append(FP_mask_maxprob)
        pred_dict_maxprob['FN'].append(FN_mask_maxprob)
        pred_dict_maxprob['TN'].append(TN_mask_maxprob)

        pred_dict_number_con['TP'].append(TP_mask_con.count_nonzero())
        pred_dict_number_con['FP'].append(FP_mask_con.count_nonzero())
        pred_dict_number_con['FN'].append(FN_mask_con.count_nonzero())
        pred_dict_number_con['TN'].append(TN_mask_con.count_nonzero())

        pred_dict_number_maxprob['TP'].append(TP_mask_maxprob.count_nonzero())
        pred_dict_number_maxprob['FP'].append(FP_mask_maxprob.count_nonzero())
        pred_dict_number_maxprob['FN'].append(FN_mask_maxprob.count_nonzero())
        pred_dict_number_maxprob['TN'].append(TN_mask_maxprob.count_nonzero())

    
        # max_prob_list = np.append(max_prob_list, estimated_max_prob.cpu().detach().numpy())
    
    
    return pred_dict_con,pred_dict_number_con,pred_dict_maxprob,pred_dict_number_maxprob #,max_prob_list

def classify_prediction(estimated_src_kps,src_kps_gt,estimated_confidence_map,pckthres,alpha,cut_off):
    l2dist = (estimated_src_kps - src_kps_gt).pow(2).sum(dim=0).pow(0.5) #torch.Size([11])
    confidence_mask_high = estimated_confidence_map.ge(cut_off) #cutoff 이상 마스크
    correct_pts = torch.le(l2dist, thres) #tensor([True, True, True, True, True, True, True, True, True, True, True],device='cuda:0') 이런식으로 bool 값 반환
    
    # correct_ids = where(correct_pts == 1) #tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10] 이런식으로 index 반환
    # incorrect_ids = where(correct_pts == 0)
    # correct_dist = l2dist[correct_pts]
    TP_mask = confidence_mask_high & correct_pts
    FP_mask = confidence_mask_high & ~correct_pts
    FN_mask = ~confidence_mask_high & correct_pts
    TN_mask = ~confidence_mask_high & ~correct_pts
    # pdb.set_trace()
    return  correct_pts,TP_mask,FP_mask,FN_mask,TN_mask # correct_ids, incorrect_ids,correct_dist,

def calculate_confusion(label_list, confidence_map, cut_off):
    mask_actual_true = torch.zeros_like(label_list)
    mask_actual_false = torch.zeros_like(label_list)
    mask_pred_true = torch.zeros_like(label_list)
    mask_pred_false = torch.zeros_like(label_list)

    mask_actual_true[label_list == True] = 1
    mask_actual_false[label_list == False] = 1

    mask_pred_true[confidence_map.ge(cut_off)] = 1
    mask_pred_false[confidence_map.le(cut_off)] = 1

    mask_TP = (mask_actual_true == True) & (mask_pred_true == True)
    mask_TN = (mask_actual_false == True) & (mask_pred_false == True)
    mask_FN = (mask_actual_true == True) & (mask_pred_false == True)
    mask_FP = (mask_actual_false == True) & (mask_pred_true == True)

    num_TP= torch.count_nonzero(mask_TP.int())
    num_TN= torch.count_nonzero(mask_TN.int())
    num_FN= torch.count_nonzero(mask_FN.int())
    num_FP= torch.count_nonzero(mask_FP.int())
    
    return (num_TP, num_TN, num_FN, num_FP)   