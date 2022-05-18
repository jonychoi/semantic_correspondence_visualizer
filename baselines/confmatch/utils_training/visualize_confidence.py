import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def visualize(mini_batch,
            ori_src_kp2D, ori_trg_kp2D,
            actual_true_src_kp2D, actual_true_tgt_kp2D,
            actual_false_src_kp2D, actual_false_tgt_kp2D,
            pred_true_src_kp2D, pred_true_tgt_kp2D,
            pred_false_src_kp2D, pred_false_tgt_kp2D,
            TP_src_kp2D, TP_tgt_kp2D,
            TN_src_kp2D, TN_tgt_kp2D,
            FN_src_kp2D, FN_tgt_kp2D,
            FP_src_kp2D, FP_tgt_kp2D,
            device, args, n_iter):
    image_ratio = 16
    args.save_img_path = os.path.join(args.save_path, 'img')
    #vis_sup_GT
    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                            mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            (ori_src_kp2D * 16, ori_trg_kp2D * 16), 
                            plot_name = 'GT_{}'.format(n_iter),
                            use_supervision ='sup', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            ori_src_kp2D * 16, ori_trg_kp2D * 16, 
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            ((actual_true_src_kp2D * 16, actual_true_tgt_kp2D * 16), 
                            (actual_false_src_kp2D * 16, actual_false_tgt_kp2D * 16)), 
                            plot_name = 'actual_{}'.format(n_iter),
                            use_supervision ='pred', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            ori_src_kp2D * 16, ori_trg_kp2D * 16, 
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            ((pred_true_src_kp2D * 16, pred_true_tgt_kp2D * 16), 
                            (pred_false_src_kp2D * 16, pred_false_tgt_kp2D * 16)), 
                            plot_name = 'pred_{}'.format(n_iter),
                            use_supervision ='pred', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            ori_src_kp2D * 16, ori_trg_kp2D * 16, 
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            ((TP_src_kp2D * 16, TP_tgt_kp2D * 16), 
                            (TN_src_kp2D * 16, TN_tgt_kp2D * 16), 
                            (FN_src_kp2D * 16, FN_tgt_kp2D * 16), 
                            (FP_src_kp2D * 16, FP_tgt_kp2D * 16)), 
                            plot_name = 'conf_{}'.format(n_iter),
                            use_supervision ='conf', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

def plot_keypoint(im_pair, src_kps, tgt_kps, src_bbox, trg_bbox,
                 pred_kps,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None, diff_idx=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]),
                            src_bbox[2] - src_bbox[0],
                            src_bbox[3] - src_bbox[1],
                            linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect_src)
    if trg_bbox is not None:
        rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]),
                                trg_bbox[2] - trg_bbox[0],
                                trg_bbox[3] - trg_bbox[1],
                                linewidth=4, edgecolor='red', facecolor='none')

        plt.gca().add_artist(rect_trg)

    if use_supervision == 'sup':
        pred_src_kps, pred_tgt_kps = pred_kps
        draw_xy(src_kps, tgt_kps, plt, c='gray')        
        draw_yx(pred_src_kps, pred_tgt_kps, plt, c='paleturquoise')

    if use_supervision == 'pred':
        true_src_kps, true_tgt_kps = pred_kps[0]
        false_src_kps, false_tgt_kps = pred_kps[1]

        draw_yx(src_kps, tgt_kps, plt, c='paleturquoise')        
        draw_yx(true_src_kps, true_tgt_kps, plt, c='green')
        draw_yx(false_src_kps, false_tgt_kps, plt, c='red')


    if use_supervision == 'conf':
        TP_src_kps, TP_tgt_kps = pred_kps[0]
        TN_src_kps, TN_tgt_kps = pred_kps[1]
        FN_src_kps, FN_tgt_kps = pred_kps[2]
        FP_src_kps, FP_tgt_kps = pred_kps[3]


        draw_yx(src_kps, tgt_kps, plt, c='paleturquoise')        
        draw_yx(TP_src_kps, TP_tgt_kps, plt, c='lightgreen')
        draw_yx(TN_src_kps, TN_tgt_kps, plt, c='deepskyblue')
        draw_yx(FN_src_kps, FN_tgt_kps, plt, c='red')
        draw_yx(FP_src_kps, FP_tgt_kps, plt, c='pink')

    elif use_supervision == 'semi':
        # if diff_idx is None:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            # c = 'coral'
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'diff':
        # else:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            c = 'red' if i in diff_idx else 'lime'
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()


def visualize_actual(mini_batch,
            actual_true_src_kp2D, actual_true_tgt_kp2D,
            actual_false_src_kp2D, actual_false_tgt_kp2D,
            device, args, n_iter,
            src_kp2D_from_16 = None, trg_kp2D_from_16 =None, plot_name=None):
    image_ratio = 16
    args.save_img_path = os.path.join(args.save_path, 'img')
    #vis_sup_GT

    #(src, trg)
    if src_kp2D_from_16 is not None:
        plot_keypoint_kps(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                                mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                                mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                                mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],                            
                                ((actual_true_src_kp2D * 16, actual_true_tgt_kp2D * 16), 
                                (actual_false_src_kp2D * 16, actual_false_tgt_kp2D * 16)),
                                mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                                src_kp2D_from_16 =  src_kp2D_from_16 * 16, trg_kp2D_from_16 =trg_kp2D_from_16 * 16, 
                                plot_name = 'actual_{}'.format(n_iter),
                                benchmark=args.benchmark, cur_snapshot=args.save_img_path)
    # (trg, trg')
    else:
        plot_keypoint_kps(torch.cat((mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),
                                    mini_batch['trg_img_strong'][0].unsqueeze(0).to(device)), 3),
                                        mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                                        mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],                            
                                        ((actual_true_src_kp2D * 16, actual_true_tgt_kp2D * 16), 
                                        (actual_false_src_kp2D * 16, actual_false_tgt_kp2D * 16)),
                                        plot_name = plot_name,
                                        benchmark=args.benchmark, cur_snapshot=args.save_img_path)        

def visualize_actual_self(mini_batch,
            actual_true_src_kp2D, actual_true_tgt_kp2D,
            actual_false_src_kp2D, actual_false_tgt_kp2D,
            device, args, n_iter):
    image_ratio = 16
    args.save_img_path = os.path.join(args.save_path, 'img')
    #vis_sup_GT


    plot_self_kps(torch.cat((mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_strong'][0].unsqueeze(0).to(device)), 3),
                            mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                            mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],                            
                            ((actual_true_src_kp2D * 16, actual_true_tgt_kp2D * 16), 
                            (actual_false_src_kp2D * 16, actual_false_tgt_kp2D * 16)), 
                            plot_name = 'actual_self_{}'.format(n_iter),
                            benchmark=args.benchmark, cur_snapshot=args.save_img_path)

def plot_keypoint_kps(im_pair, anno_src, anno_tgt, pred_kps,
                    src_bbox = None, trg_bbox = None, src_kp2D_from_16 = None, trg_kp2D_from_16 = None,
                 plot_name=None, use_supervision=None, benchmark=None, cur_snapshot=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)
    if src_bbox is not None:
        rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]),
                                src_bbox[2] - src_bbox[0],
                                src_bbox[3] - src_bbox[1],
                                linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_artist(rect_src)
    if trg_bbox is not None:
        rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]),
                                trg_bbox[2] - trg_bbox[0],
                                trg_bbox[3] - trg_bbox[1],
                                linewidth=2, edgecolor='red', facecolor='none')
        plt.gca().add_artist(rect_trg)

    true_src_kps, true_tgt_kps = pred_kps[0]
    false_src_kps, false_tgt_kps = pred_kps[1]
    if src_kp2D_from_16 is not None and trg_kp2D_from_16 is not None:
        draw_xy(anno_src, anno_tgt, plt, c='black')        
        draw_xy(src_kp2D_from_16, trg_kp2D_from_16, plt, c='c')        
        draw_xy(true_src_kps, true_tgt_kps, plt, c='green')
        draw_xy(false_src_kps, false_tgt_kps, plt, c='red')
    else:
        draw_yx(true_src_kps, true_tgt_kps, plt, c='green')
        draw_yx(false_src_kps, false_tgt_kps, plt, c='red')
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()

def plot_self_kps(im_pair, anno_src, anno_tgt, 
                 pred_kps,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None, diff_idx=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)


    true_src_kps, true_tgt_kps = pred_kps[0]
    false_src_kps, false_tgt_kps = pred_kps[1]
    draw_xy(anno_src, anno_tgt, plt, c='black')        
    draw_yx(true_src_kps, true_tgt_kps, plt, c='green')
    draw_yx(false_src_kps, false_tgt_kps, plt, c='red')

    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()

def plot_image(im, return_im=True):
    mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1))
    std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1))
    if im.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    im = im.mul(std).add(mean) * 255.0
    im = im.data.squeeze(0).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im
    plt.imshow(im)
    plt.show()
def draw_xy(src_kps, tgt_kps, plt, c=None):
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[0, i]) + 256
            yb = float(tgt_kps[1, i])
            if c is None:
                c = np.random.rand(3)            
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)

def draw_yx(src_kps, tgt_kps, plt, c=None):
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            if c is None:
                c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)

def plot_keypoint(im_pair, src_kps, tgt_kps, src_bbox, trg_bbox,
                 pred_kps,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None, diff_idx=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]),
                            src_bbox[2] - src_bbox[0],
                            src_bbox[3] - src_bbox[1],
                            linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect_src)
    if trg_bbox is not None:
        rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]),
                                trg_bbox[2] - trg_bbox[0],
                                trg_bbox[3] - trg_bbox[1],
                                linewidth=4, edgecolor='red', facecolor='none')

        plt.gca().add_artist(rect_trg)

    if use_supervision == 'sup':
        pred_src_kps, pred_tgt_kps = pred_kps
        draw_xy(src_kps, tgt_kps, plt, c='gray')        
        draw_yx(pred_src_kps, pred_tgt_kps, plt, c='paleturquoise')

    if use_supervision == 'pred':
        true_src_kps, true_tgt_kps = pred_kps[0]
        false_src_kps, false_tgt_kps = pred_kps[1]

        draw_yx(src_kps, tgt_kps, plt, c='paleturquoise')        
        draw_yx(true_src_kps, true_tgt_kps, plt, c='green')
        draw_yx(false_src_kps, false_tgt_kps, plt, c='red')

    if use_supervision == 'conf':
        TP_src_kps, TP_tgt_kps = pred_kps[0]
        TN_src_kps, TN_tgt_kps = pred_kps[1]
        FN_src_kps, FN_tgt_kps = pred_kps[2]
        FP_src_kps, FP_tgt_kps = pred_kps[3]


        draw_yx(src_kps, tgt_kps, plt, c='paleturquoise')        
        draw_yx(TP_src_kps, TP_tgt_kps, plt, c='lightgreen')
        draw_yx(TN_src_kps, TN_tgt_kps, plt, c='deepskyblue')
        draw_yx(FN_src_kps, FN_tgt_kps, plt, c='red')
        draw_yx(FP_src_kps, FP_tgt_kps, plt, c='pink')

    elif use_supervision == 'semi':
        # if diff_idx is None:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            # c = 'coral'
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'diff':
        # else:
        for i in range(src_kps.size(1)):
            xa = float(src_kps[1, i])
            ya = float(src_kps[0, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            c = 'red' if i in diff_idx else 'lime'
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()

# Working
def plot_diff_keypoint(im_pair, src_kps, tgt_kps, src_bbox,
                 plot_name, use_supervision=None, benchmark=None, cur_snapshot=None):
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect = plt.Rectangle((src_bbox[0], src_bbox[1]),
                            src_bbox[2] - src_bbox[0],
                            src_bbox[3] - src_bbox[1],
                            linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect)
    if use_supervision == 'sup':
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[0, i]) + 256
            yb = float(tgt_kps[1, i])
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    elif use_supervision == 'semi':
        for i in range(src_kps.size(1)):
            xa = float(src_kps[0, i])
            ya = float(src_kps[1, i])
            xb = float(tgt_kps[1, i]) + 256
            yb = float(tgt_kps[0, i])
            c = np.random.rand(3)
            plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
            plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
            plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
    save_dir = f'{cur_snapshot}'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name),
    bbox_inches='tight')
    plt.close()