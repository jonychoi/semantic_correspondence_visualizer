from email.policy import strict
import re
import os
import shutil

import torch
import torch.nn.functional as F
import numpy as np
import pdb
from collections import OrderedDict
r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def save_checkpoint(state, is_best, save_path, filename=None):
    # torch.save(state, os.path.join(save_path,filename))
    if is_best:    
        if filename:
            torch.save(state, os.path.join(save_path, filename))
        else:
            torch.save(state, os.path.join(save_path, 'model_best.pth'))

        # shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth'))
def save_checkpoint_per_epoch(state, save_path,epoch, filename):
    
    torch.save(state, os.path.join(save_path,f"model_best_{epoch}.pth"))
def load_checkpoint_resume(model, optimizer,optimizer_con, scheduler,scheduler_con, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location='cpu')
        new_state_dict = OrderedDict()
        print(checkpoint['state_dict'].keys())
        for n, v in checkpoint['state_dict'].items():
            if n.startswith('module'):
                new_n = n[7:]
                new_state_dict[new_n] = v

            else:
                new_state_dict[n] = v        
        
        start_epoch = checkpoint['epoch']
        
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer_con.load_state_dict(checkpoint['optimizer_con'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scheduler_con.load_state_dict(checkpoint['scheduler_con'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer,optimizer_con, scheduler,scheduler_con, start_epoch, best_val

def load_checkpoint_conf(model, optimizer,optimizer_con, scheduler,scheduler_con, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cpu')
        new_state_dict = OrderedDict()
        print(checkpoint['state_dict'].keys())
        for n, v in checkpoint['state_dict'].items():
            if n.startswith('con_estimator'):
                new_state_dict[n] = v
            # elif n.startswith('x_normal') or n.startswith('y_normal'):
            #     new_n = 'net.{}'.format(n)
            #     new_state_dict[n] = v
            #     # new_state_dict[new_n] = v
            else:
                new_n = 'net.{}'.format(n)
                new_state_dict[new_n] = v

        missing_keys = model.load_state_dict(new_state_dict, strict = False)
        print(missing_keys)        
        start_epoch = checkpoint['epoch']
        # model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer_con.load_state_dict(checkpoint['optimizer_con'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        # scheduler_con.load_state_dict(checkpoint['scheduler_con'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer,optimizer_con, scheduler,scheduler_con, start_epoch, best_val

r'''
    source code from GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''
def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    best_val=-1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        checkpoint['scheduler']['milestones'] = scheduler.__dict__['milestones']  
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        try:
            best_val=checkpoint['best_loss']
        except:
            best_val=-1
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, scheduler, start_epoch, best_val


# Argument parsing
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def log_args(args):
    r"""Log program arguments"""
    print('\n+================================================+')
    for arg_key in args.__dict__:
        print('| %20s: %-24s |' % (arg_key, str(args.__dict__[arg_key])))
    print('+================================================+\n')


def parse_list(list_str):
    r"""Parse given list (string -> int)"""
    return list(map(int, re.findall(r'\d+', list_str)))


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
