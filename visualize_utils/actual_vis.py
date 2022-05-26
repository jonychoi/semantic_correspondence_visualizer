import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
import os

def actual_src_tgt_kps(i, mini_batch, args, device):
        
    src = mini_batch['src_img'][0].unsqueeze(0).to(device)
    tgt = mini_batch['trg_img'][0].unsqueeze(0).to(device)
    src_kps = mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]]
    tgt_kps = mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]]

    plot_name_tgt = 'target_{}'.format(i)
    plot_name_src = 'source_{}'.format(i)

    imgs = [tgt, src]
    names = [plot_name_tgt, plot_name_src]
    kps = [tgt_kps, src_kps]
    c = ['red', 'aqua']
    
    for n, img in enumerate(zip(imgs, names, kps, c)):
        _img = plot_image(img[0], return_im=True)
        plt.imshow(_img)
        
        for i in range(img[2].size(1)):
            x = float(img[2][0, i])
            y = float(img[2][1, i])

            c = np.random.rand(3)
            
            plt.gca().add_artist(plt.Circle((x, y), radius=5, color=img[3]))
            plt.axis('off')

        if n == 0:
            save_dir = args.save_dir + '/GT/{}/{}'.format(args.dataset, 'target')
        elif n == 1:
            save_dir = args.save_dir + '/GT/{}/{}'.format(args.dataset, 'source')
        os.makedirs(save_dir, exist_ok = True)
        if n == 0:
            plt.savefig('{}/{}.png'.format(save_dir, img[1]), bbox_inches='tight')
        elif n == 1:
            plt.savefig('{}/{}.png'.format(save_dir, img[1]), bbox_inches='tight')
        plt.close()

def actual_src_tgt_mapping(i, mini_batch, args, device):
    im_pair = torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device), mini_batch['trg_img'][0].unsqueeze(0).to(device)), 3)
    src_kps = mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]]
    tgt_kps = mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]]
    src_bbox = mini_batch['src_bbox'][0]
    trg_bbox = mini_batch['trg_bbox'][0]

    plot_name = 'mapping_{}'.format(i)
    im = plot_image(im_pair, return_im=True)
    plt.imshow(im)

    rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]), src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1], linewidth=4, edgecolor='red', facecolor='none')
    plt.gca().add_artist(rect_src)

    if trg_bbox is not None:
        rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]), trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1], linewidth=4, edgecolor='red', facecolor='none')
        plt.gca().add_artist(rect_trg)

    for i in range(src_kps.size(1)):
        xa = float(src_kps[0, i])
        ya = float(src_kps[1, i])
        xb = float(tgt_kps[0, i]) + 256
        yb = float(tgt_kps[1, i])
        c = np.random.rand(3)
        plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
        plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
        plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)
        plt.axis('off')

    save_dir = args.save_dir + '/GT/{}/mapping'.format(args.dataset)
    os.makedirs(save_dir, exist_ok = True)
    print(save_dir)
    plt.savefig('{}/{}.png'.format(save_dir, plot_name), bbox_inches='tight')
    plt.close()

def plot_image(im, return_im=True):
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    if im.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    im = im.mul(std).add(mean) * 255.0
    im = im.data.squeeze(0).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
    if return_im:
        return im