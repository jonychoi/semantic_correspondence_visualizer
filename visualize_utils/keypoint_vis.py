import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

def where(predicate):
    r"""Predicate must be a condition on nd-tensor"""
    matching_indices = predicate.nonzero()
    if len(matching_indices) != 0:
        matching_indices = matching_indices.t().squeeze(0)
    return matching_indices


def flow2kps(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()
    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
        size = trg_kps.size(1)

        kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts).cuda() * -1), dim=1)
        src_kps.append(estimated_kps)

    return torch.stack(src_kps)


def classify_prd(cls, prd_kps, trg_kps, pckthres):
    r"""Compute the number of correctly transferred key-points"""
    l2dist = (prd_kps - trg_kps).pow(2).sum(dim=0).pow(0.5)
    thres = pckthres.expand_as(l2dist).float() * cls.alpha
    correct_pts = torch.le(l2dist, thres)

    correct_ids = where(correct_pts == 1)
    incorrect_ids = where(correct_pts == 0)
    correct_dist = l2dist[correct_pts]

    return correct_dist, correct_ids, incorrect_ids

def eval_kps_transfer_with_correct(cls, prd_kps, batch):
    r"""Compute percentage of correct key-points (PCK) based on prediction"""

    easy_match = {'src': [], 'trg': [], 'dist': []}
    hard_match = {'src': [], 'trg': []}

    pck = []
    correct_id_list = []
    for idx, (pk, tk) in enumerate(zip(prd_kps, batch['src_kps'])):
        thres = batch['pckthres'][idx]
        npt = batch['n_pts'][idx]
        correct_dist, correct_ids, incorrect_ids = cls.classify_prd(pk[:, :npt], tk[:, :npt], thres)
        correct_id_list.append(correct_ids)

        pck.append((len(correct_ids) / npt.item()) * 100)

    eval_result = {'pck': pck}

    return eval_result, correct_id_list




def keypointer(args, model_name, index, save_dir, theme, name_plot = False, name = None):
    true_kpts = []