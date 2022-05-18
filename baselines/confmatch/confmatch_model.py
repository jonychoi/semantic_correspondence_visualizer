import random
import yaml
import sys
import numpy as np
import torch
sys.path.append('.')
from models.cats import CATs
from confidence_estimator.con_estimator_tf_not_residual import con_estimator_tf_not_residual_shallow
from semimatch.utils import parse_list
from confmatch.confmatch import ConfMatch

class args_struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

with open('finalargs.yaml') as f:
    args_yaml = yaml.load(f)
args = args_struct(**args_yaml)
        
def confmatch(dataset):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dataset == 'pfpascal' or 'pfwillow':
        proj_feat_input_dim = 896
    elif dataset == 'spair' :
        proj_feat_input_dim = 1024

    # Model
    model = CATs(feature_proj_dim=args.feature_proj_dim, depth=1, num_heads=6, mlp_ratio=4, hyperpixel_ids=parse_list(args.hyperpixel), freeze=True, args=args)
    confidence_estimator = con_estimator_tf_not_residual_shallow(proj_feat_input_dim = proj_feat_input_dim, depth=args.con_est_depth)
    confmatch = ConfMatch(model, confidence_estimator, device, args)
    confmatch = confmatch.to(device)

    return confmatch