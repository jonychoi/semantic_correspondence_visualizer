import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

def EPE(input_flow, target_flow, sparse=True, mean=True, sum=False, mask=None, weight=None):
    
    EPE_map = torch.norm(target_flow-input_flow, 2, 1)
    batch_size = EPE_map.size(0)
    H= EPE_map.size(1)
    W= EPE_map.size(2)

    if weight is not None :
        EPE_map = weight.cuda().repeat(H*W).reshape(batch_size,H,W).mul(EPE_map)

    # Sup loss
    if mask is None:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    
    # mask_2D unsup loss
    else:
        EPE_map = EPE_map[mask]

    if mean:
        return EPE_map.mean()
    elif sum:
        return EPE_map.sum()
    else:
        return EPE_map.sum()/torch.sum(mask)


def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    criterion = nn.BCELoss()
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        return criterion(logits, targets)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss



def consistency_loss(prob_w, prob_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    prob_w = prob_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        # pseudo_label = torch.softmax(logits_w, dim=-1)
        # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mask = prob_w.ge(p_cutoff).float()
        prob_s_1D= prob_s.view(-1)
        # zero_1D = torch.zeros_like(prob_s_1D, device=device, dtype=torch.float16)
        # prob_s = torch.cat((prob_s_1D, zero_1D), dim=1)
        # labels = torch.ones(int(prob_s_1D.size(0)), device=device, dtype=torch.float32)
        mask1D = prob_w.ge(p_cutoff).float().view(-1)

        if use_hard_labels:
            masked_loss = ce_loss(prob_s_1D, mask1D, use_hard_labels, reduction='none')
        # else:
        #     pseudo_label = torch.softmax(logits_w / T, dim=-1)
        #     masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss, mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def contrastive_loss(loss, corr_strong, index_weak_transformed, mask_weak_transformed, device=None, use_loss_weight=False, contrastive_temp=None):
    B, Svec, T_h, T_w = corr_strong.size()
    masked_corr_strong = corr_strong.permute(0,2,3,1).reshape(B*T_h *T_w, Svec)
    masked_corr_strong = masked_corr_strong[mask_weak_transformed.view(-1)]
    
    masked_index_weak_transformed = index_weak_transformed[mask_weak_transformed].long()
    masked_num = masked_index_weak_transformed.size(0)    

    semi_loss_weight = torch.ones(B*T_h*T_w)


    if not masked_num == 0:
        mask_pixelCT = torch.zeros_like(masked_corr_strong).bool()
        mask_pixelCT[torch.arange(masked_num), masked_index_weak_transformed] = True
        positive = masked_corr_strong[mask_pixelCT].view(masked_num, -1)
        negative = masked_corr_strong[~mask_pixelCT].view(masked_num, -1)
        masked_pred = torch.cat([positive, negative], dim=1)
        
        masked_labels = torch.zeros(int(masked_num), device=device, dtype=torch.int64)
        masked_labels = masked_labels.detach()
        eps_temp = 1e-6
        masked_pred_with_temp = (masked_pred / contrastive_temp) + eps_temp
        if use_loss_weight == False:
            loss.reduction = 'mean'
            loss_unsup = loss(masked_pred_with_temp, masked_labels)
    else:
        loss_unsup = torch.tensor(0.0, device=device)
    return loss_unsup        