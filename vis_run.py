import sys, time, argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
from collections import OrderedDict
from data import download
from visualize_utils.predictors import predict_confmatch, predict_cat

#import models
from baselines.cats.models.cats import CATs as cats
from baselines.confmatch.confmatch_model import confmatch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_threads', type=int, default=32)
parser.add_argument('--thres', type=str, default='auto')
parser.add_argument('--datapath', type=int, default='../../../Datasets_CATs')
parser.add_argument('--dataset', type=str, default='pfpascal')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
device = torch.cuda.current_device()

test_dataset = download.load_dataset(args.dataset, args.datapath, args.thres, device, 'test', False, 16)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False)
    
vis_started_time = time.time()

#Q. store_true => true
#Q. cats what to use?? =>target

#models
confmatch = confmatch(args.dataset)

models = [confmatch, cats]
pre_trained_weights = [args.confmatch_pretrained_path, args.cats_pretrained_path, args.chm_pretrained_path, args.semimatch_pretrained_path]

#get model sota weights and predict the confidence masks or keypoint mapping
for index, model in enumerate(models):
    checkpoint = torch.load(pre_trained_weights[index], map_location='cpu')
    new_state_dict = OrderedDict()
    for n, v in checkpoint['state_dict'].items():
        if n.startswith('module'):
            new_n = n[7:]
            new_state_dict[new_n] = v
        else:
            new_state_dict[n] = v     
    print(new_state_dict.keys())
    model.load_state_dict(new_state_dict)

    model.eval()

    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
        
        for i, mini_batch in pbar:
            if model == confmatch:
                predict_confmatch(model, mini_batch, args, device)
            elif model == cats:
                predict_cat(model, mini_batch, args, device)



print(args.seed, 'Visualize time took: ', time.time()-vis_started_time, 'seconds')