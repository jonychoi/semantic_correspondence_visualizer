import random
import yaml
import os
import numpy as np
import torch
from baselines.confmatch.models.cats import CATs
from baselines.confmatch.confidence_estimator.con_estimator_tf_not_residual import con_estimator_tf_not_residual_shallow
from baselines.confmatch.utils_training.utils import parse_list
from baselines.confmatch.confmatch.confmatch import ConfMatch

class args_struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

with open(os.getcwd() + '/baselines/confmatch/finalargs.yml') as f:
    args_yaml = yaml.load(f)
args = args_struct(**args_yaml)
        
def confmatch(dataset):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'pfpascal' or dataset == 'pfwillow':
        proj_feat_input_dim = 896
        hyperpixel_ids = [2,17,21,22,25,26,28]
    elif dataset == 'spair' :
        proj_feat_input_dim = 1024
        hyperpixel_ids = [0,8,20,21,26,28,29,30]

    # Model
    model = CATs(feature_proj_dim=args.feature_proj_dim, depth=1, num_heads=6, mlp_ratio=4, hyperpixel_ids=hyperpixel_ids, freeze=True, args=args)
    confidence_estimator = con_estimator_tf_not_residual_shallow(proj_feat_input_dim = proj_feat_input_dim, depth=args.con_est_depth)
    confmatch = ConfMatch(model, confidence_estimator, device, args)
    confmatch = confmatch.to(device)

    return confmatch