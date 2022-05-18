r'''
    modified test script of GLU-Net
    https://github.com/PruneTruong/GLU-Net
'''

import argparse
import os
import pickle
import random
import time
from os import path as osp
import yaml
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from termcolor import colored
from torch.utils.data import DataLoader
from confidence_estimator.con_estimator_tf_not_residual import con_estimator_tf_not_residual_shallow
from confidence_estimator.con_estimator_sh import con_estimator_sh

sys.path.append('.')
from models.cats import CATs
import utils_training.optimize_semimatch as optimize
from semimatch.evaluation import Evaluator
from semimatch.utils import parse_list, log_args, load_checkpoint, save_checkpoint, boolean_string
from confmatch.utils import load_checkpoint_resume
from confmatch.confmatch import ConfMatch
import utils_training.optimize_confmatch as optimize_confmatch

from data import download
from collections import OrderedDict
import pdb
class args_struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
        
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CATs Test Script')
    # Paths
    parser.add_argument('--name_exp', type=str,
                        default=time.strftime('%Y_%m_%d_%H_%M'),
                        help='name of the experiment to save')
    parser.add_argument('--snapshots', type=str, default='./eval')
    parser.add_argument('--pretrained', dest='pretrained',
                       help='path to pre-trained model')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='training batch size')
    parser.add_argument('--n_threads', type=int, default=32,
                        help='number of parallel threads for dataloaders')
    parser.add_argument('--seed', type=int, default=2021,
                        help='Pseudo-RNG seed')
                        
    parser.add_argument('--datapath', type=str, default='../Datasets_CATs')
    parser.add_argument('--benchmark', type=str, choices=['pfpascal', 'spair', 'pfwillow'])
    parser.add_argument('--thres', type=str, default='auto', choices=['auto', 'img', 'bbox'])
    parser.add_argument('--alpha', type=float, default=0.1)

    #confidence_estimator
    parser.add_argument('--con_est_type', type=str, default='2D', choices=['2D', '4D', 'tf', 'tf_proj', 'maxProb', 'tf_not_residual', 'tf_not_residual_shallow', 'tf_not_residual_shallow_mean','SH'])
    parser.add_argument('--con_est_depth', type=int, default=1)
    parser.add_argument('--sparse_exp', action='store_true')
    parser.add_argument('--semi_softmax_corr_temp', type=float, default=0.1)

    # Seed
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize Evaluator
    Evaluator.initialize(args.benchmark, args.alpha)
    
    # with open(osp.join(args.pretrained, 'args.pkl'), 'rb') as f:
    #     args_model = pickle.load(f)
    # log_args(args_model)
    with open(osp.join(args.pretrained, 'args.yaml'), 'rb') as f:
        args_yaml = yaml.load(f)
    args_model = args_struct(**args_yaml)
    

    # Dataloader
    download.download_dataset(args.datapath, args.benchmark)
    test_dataset = download.load_dataset(args.benchmark, args.datapath, args.thres, device, 'test', False, args_model.feature_size)
    test_dataloader = DataLoader(test_dataset,
        batch_size=args.batch_size,
        num_workers=args.n_threads,
        shuffle=False)

    # Model
    model = CATs(
        feature_size=args_model.feature_size, feature_proj_dim=args_model.feature_proj_dim,
        depth=args_model.depth, num_heads=args_model.num_heads, mlp_ratio=args_model.mlp_ratio,
        hyperpixel_ids=parse_list(args_model.hyperpixel), freeze=True, args=args_model)

    if args.benchmark == 'pfpascal' or 'pfwillow':
        proj_feat_input_dim = 896
    elif args.benchmark == 'spair' :
        proj_feat_input_dim = 1024


    if args.con_est_type == 'tf_not_residual_shallow':
        confidence_estimator = con_estimator_tf_not_residual_shallow(proj_feat_input_dim = proj_feat_input_dim, depth=args.con_est_depth)
    elif args.con_est_type == 'SH':
        confidence_estimator = con_estimator_sh(depth=args.con_est_depth, proj_feat_input_dim = proj_feat_input_dim) 

    confmatch = ConfMatch(model, confidence_estimator, device, args)

    if args.pretrained:
        checkpoint = torch.load(osp.join(args.pretrained, 'model_best.pth'), map_location='cpu')
        new_state_dict = OrderedDict()
        for n, v in checkpoint['state_dict'].items():
            if n.startswith('module'):
                new_n = n[7:]
                new_state_dict[new_n] = v

            else:
                new_state_dict[n] = v     
        print(new_state_dict.keys())
        confmatch.load_state_dict(new_state_dict)
    else:
        raise NotImplementedError()
    # create summary writer

    # model = nn.DataParallel(model)
    model = model.to(device)

    train_started = time.time()

    # val_loss_grid, val_mean_pck, mean_pck_cat_array = optimize.validate_epoch_test(model,
    #                                                 test_dataloader,
    #                                                 device,
    #                                                 epoch=0)
    val_loss_grid, val_mean_pck = optimize.validate_epoch(model,
                                                    test_dataloader,
                                                    device,
                                                    epoch=0)
    print(colored('==> ', 'blue') + 'Test average grid loss :',
            val_loss_grid)
    print('mean PCK is {}'.format(val_mean_pck))
    # print("category pck")
    # print(mean_pck_cat_array)

    print(args.seed, 'Test took:', time.time()-train_started, 'seconds')
