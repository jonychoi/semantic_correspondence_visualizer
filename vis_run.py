import time
import argparse
import random
import numpy as np
from visualize_utils.actual_vis import actual_src_tgt_kps, actual_src_tgt_mapping
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import OrderedDict
from data import download
from visualize_utils import predictors

#import models
from baselines.cats.cats_model import Cats
from baselines.confmatch.confmatch_model import confmatch as Confmatch
from visualize_utils.tgt_test_imgs import pfpascal, spair

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="visualization")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=32)
    parser.add_argument('--thres', type=str, default='auto')
    parser.add_argument('--datapath', type=str, default='../../../Datasets_CATs')
    parser.add_argument('--dataset', type=str, default='pfpascal')
    parser.add_argument('--confmatch_pretrained_path', type=str, default='../model_weights/confmatch_pascal_best.pth')
    parser.add_argument('--cats_pretrained_path', type=str, default='../model_weights/cats_pascal_best.pth')
    parser.add_argument('--cats_pretrained_path_spair', type=str, default='../model_weights/cats_spair_best.pth')
    parser.add_argument('--confmatch_pretrained_path_spair', type=str, default='../model_weights/confmatch_spair_best.pth')
    parser.add_argument('--kps_or_mask', type=str, default='mask')
    parser.add_argument('--save_dir', type=str, default='./imgs')
    parser.add_argument('--seed', type=int, default='1998')
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--image_opacity', type=float, default=0.55)
    parser.add_argument('--first_masking_order', type=str, default="mask")
    parser.add_argument('--gt', type=str, default='no')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = download.load_dataset(args.dataset, args.datapath, args.thres, device, 'test', False, 16)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads, shuffle=False)
        
    vis_started_time = time.time()

    #Q. store_true => true
    #Q. cats what to use?? =>target

    #models
    print(args.dataset)
    confmatch = Confmatch(args.dataset)
    cats = Cats(args.dataset)

    models = [confmatch, cats]

    global tst_anno
    
    if args.dataset == 'pfpascal':
        tst_anno = pfpascal()
        pre_trained_weights = [args.confmatch_pretrained_path, args.cats_pretrained_path]
    elif args.dataset == 'spair':
        tst_anno = spair()
        pre_trained_weights = [args.confmatch_pretrained_path_spair, args.cats_pretrained_path_spair]
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
        #print(new_state_dict.keys())
        model.load_state_dict(new_state_dict, strict = False)

        model.eval()

        with torch.no_grad():
            pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
            
            for i, mini_batch in pbar:

                if args.gt == 'okay':
                    #actual_src_tgt_mapping(i, mini_batch, args, device)
                    actual_src_tgt_kps(i, mini_batch, args, device)
                else:
                    if model == confmatch:
                        predictors.predict_confmatch(i, model, mini_batch, args, device, tst_anno)
                        pass
                    elif model == cats:
                        predictors.predict_cat(i, model, mini_batch, args, device, tst_anno)



    print(args.seed, 'Visualize time took: ', time.time()-vis_started_time, 'seconds')