import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from data.dataset import TpsGridGen
import os
import numpy as np

#optimize from confmatch

def keypointer(args, model_name, net, index, mini_batch, save_dir, theme, name_plot = False, name = None):
    #corr: softmax_with_temperatured_corr_T_Svec (3D)
    #weak
    _, corr_weak, _, occ_T_Svec= net(mini_batch['trg_img_weak'].to(device), mini_batch['src_img'].to(device), vis_fbcheck=True)
    
    B, corrdim, W, H = corr_weak.size(0), args.feature_size * args.feature_size, args.feature_size, args.feature_size

    #not_transformed_weak
    # corr_weak = corr_weak / args.softmax_corr_temp

    corr_weak = corr_weak.view(B, corrdim, H, W)
    corr_weak_prob = net.softmax_with_temperature(corr_weak, args.semi_softmax_corr_temp)

    score_weak, index_weak = torch.max(corr_weak_prob, dim=1)
    x, y = (index_weak % W), torch.div(index_weak, W, rounding_mode='floor')

    score_mask = score_weak.ge(args.p_cutoff) #(B, 16, 16)

    bbox = torch.round(mini_batch['src_bbox'] / args.feature_size).cuda().long() # B x (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]

    src_bbox_mask = (x >= x1.repeat_interleave(256).view(-1, W, W)) & \
                    (x <= x2.repeat_interleave(256).view(-1, W, W)) & \
                    (y >= y1.repeat_interleave(256).view(-1, W, W)) & \
                    (y <= y2.repeat_interleave(256).view(-1, W, W))

    bbox = torch.round(mini_batch['trg_bbox'] / args.feature_size).cuda().long() # B x (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox[:,0], bbox[:,1], bbox[:,2], bbox[:,3]
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, H, W).repeat(B, 1, 1).cuda()
    yy = yy.view(1, H, W).repeat(B, 1, 1).cuda()

    trg_bbox_mask = (xx >= x1.repeat_interleave(256).view(-1, W, W)) & \
                    (xx <= x2.repeat_interleave(256).view(-1, W, W)) & \
                    (yy >= y1.repeat_interleave(256).view(-1, W, W)) & \
                    (yy <= y2.repeat_interleave(256).view(-1, W, W))
    #vis_variable(weak_not_T)
    mask2D_not_T = score_mask & src_bbox_mask & trg_bbox_mask
    mask2D_not_T = mask2D_not_T & occ_T_Svec.squeeze(1).bool()

    mask_tgt_kp2D_weak_not_T = (mask2D_not_T[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
    mask_src_1D = index_weak[0][mask2D_not_T[0]]
    mask_src_kp2D_weak_not_T = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

    # transformed_weak
    x_transformed, y_transformed = torch.round( transform_by_grid(x.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode, interpolation_mode=args.interpolation_mode)).squeeze().long().clamp(0, 255), torch.round(transform_by_grid(y.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode, interpolation_mode=args.interpolation_mode)).squeeze().long().clamp(0, 255)
    index_weak = y_transformed * W + x_transformed

    mask_2D = torch.round(transform_by_grid(mask2D_not_T.unsqueeze(1).float(), mini_batch[args.aug_mode].to(device), mode=args.aug_mode, interpolation_mode=args.interpolation_mode)).squeeze().bool()
    #vis_variable(weak_T)
    mask_tgt_kp2D_weak = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
    mask_src_1D = index_weak[0][mask_2D[0]]
    mask_src_kp2D_weak = torch.cat(((mask_src_1D % W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

    #strong#
    mini_batch['trg_img_strong'] = transform_by_grid(mini_batch['trg_img_strong'].to(device), mini_batch[args.aug_mode].to(device),  mode=args.aug_mode) 
    _, corr_strong, _, _ = net(mini_batch['trg_img_strong'].to(device), mini_batch['src_img'].to(device))

    corr_strong = corr_strong.view(B, corrdim, H, W)
    corr_strong_prob = net.softmax_with_temperature(corr_strong, args.semi_softmax_corr_temp)
    score_strong, index_strong = torch.max(corr_strong_prob, dim=1)
    mask_tgt_kp2D_strong = (mask_2D[0] == True).nonzero(as_tuple=False).transpose(-1, -2)
    mask_src_1D = index_strong[0][mask_2D[0]]
    mask_src_kp2D_strong = torch.cat(((mask_src_1D %  W).view(1,-1), (mask_src_1D // W).view(1,-1)), dim=0)

    GPU_NUM = torch.cuda.current_device()
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    #(corr_T_Svec: source coord, tgt_vec) corr.reshape (B, S, S, S*S)


    image_ratio = 16
    #vis_sup_GT
    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]],
                            mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]],
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            plot_name = 'GT_{}'.format(n_iter),
                            use_supervision ='sup', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_weak'][0].unsqueeze(0).to(device)), 3),
                            mask_src_kp2D_weak_not_T * image_ratio,
                            mask_tgt_kp2D_weak_not_T * image_ratio,
                            mini_batch['src_bbox'][0], mini_batch['trg_bbox'][0],
                            plot_name = 'weak_not_T_{}'.format(n_iter),
                            use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    #vis_unsup
    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            # affine_transform(mini_batch['trg_img_weak'][0].unsqueeze(0),  mini_batch['aff'][0]).to(device)), 3),
                            transform_by_grid(mini_batch['trg_img_weak'][0].unsqueeze(0).to(device),  mini_batch[args.aug_mode][0].unsqueeze(0).to(device), mode=args.aug_mode)),3),
                            mask_src_kp2D_weak * image_ratio,
                            mask_tgt_kp2D_weak * image_ratio,
                            mini_batch['src_bbox'][0], None,
                            plot_name = 'weak_{}'.format(n_iter),
                            use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)

    plot_keypoint(torch.cat((mini_batch['src_img'][0].unsqueeze(0).to(device),
                            mini_batch['trg_img_strong'][0].unsqueeze(0).to(device)), 3),
                            mask_src_kp2D_strong * image_ratio,
                            mask_tgt_kp2D_strong * image_ratio,
                            mini_batch['src_bbox'][0], None,
                            plot_name = 'strong_{}'.format(n_iter),
                            use_supervision ='semi', benchmark=args.benchmark, cur_snapshot=args.save_img_path)


def transform_by_grid(src:torch.Tensor, theta=None, mode='aff', interpolation_mode = 'bilinear', padding_factor=1.0, crop_factor=1.0, use_mask=True):
    mode_list = []
    if mode == 'aff' or mode == 'tps':
        mode_list=[mode]
        theta_list = [theta]
    if mode == 'afftps':
        mode_list = ['aff', 'tps']
        theta_list = [theta[:,:6], theta[:,6:]]
    for i in range(len(mode_list)):
        theta = theta_list[i].float()
        sampling_grid = generate_grid(src.size(), theta, mode=mode_list[i])
        sampling_grid = sampling_grid.cuda()
        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        src = F.grid_sample(src, sampling_grid.float(), align_corners=False, mode=interpolation_mode)
        mask = torch.autograd.Variable(torch.ones(src.size())).cuda()
        if use_mask:
            mask = F.grid_sample(mask, sampling_grid)
            mask[mask < 0.9999] = 0
            mask[mask > 0] = 1
            src = mask*src
    return src


#Return transformed grid#
def generate_grid(img_size, theta=None, mode='aff'):
    out_h, out_w = img_size[2], img_size[3]
    gridGen = TpsGridGen(out_h, out_w)

    if mode == 'aff':
        return F.affine_grid(theta.view(-1,2,3), img_size)
    elif mode == 'tps':
        return gridGen(theta.view(-1,18,1,1))
    else:
        raise NotImplementedError

# def keypointer(args, model_name, index, mini_batch, save_dir, theme, name_plot = False, name = None):

#     im_pair = torch.cat((mini_batch['src_img'][0].unsqueeze(0), mini_batch['trg_img_weak'][0].unsqueeze(0)), 3)
#     src_kps = mini_batch['src_kps'][0][:, 0:mini_batch['n_pts'][0]]
#     tgt_kps = mini_batch['trg_kps'][0][:, 0:mini_batch['n_pts'][0]]
#     src_bbox = mini_batch['src_bbox'][0]
#     trg_bbox = mini_batch['trg_bbox'][0]

#     image = plot_image(im_pair)

#     plt.imshow(image)

#     if name_plot:
#         plt.title(name, size=14)

#     rect_src = plt.Rectangle((src_bbox[0], src_bbox[1]), src_bbox[2] - src_bbox[0], src_bbox[3] - src_bbox[1], linewidth=4, edgecolor='red', facecolor='none')
#     plt.gca().add_artist(rect_src)

#     if trg_bbox is not None:
#         rect_trg = plt.Rectangle((trg_bbox[0] + 256, trg_bbox[1]), trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1], linewidth=4, edgecolor='red', facecolor='none')

#         plt.gca().add_artist(rect_trg)

#     for i in range(src_kps.size(1)):
#         xa = float(src_kps[0, i])
#         ya = float(src_kps[1, i])
#         xb = float(tgt_kps[0, i]) + 256
#         yb = float(tgt_kps[1, i])
#         c = np.random.rand(3)
#         plt.gca().add_artist(plt.Circle((xa, ya), radius=3, color=c))
#         plt.gca().add_artist(plt.Circle((xb, yb), radius=3, color=c))
#         plt.plot([xa, xb], [ya, yb], c=c, linestyle='-', linewidth=1.0)

#     save_plot(dir_name = save_dir, img_name="{}'s {} keypoints mapping of {}".format(args.dataset, index, model_name))

# def plot_image(image):
#     mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#     std = torch.FloatTensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#     if image.is_cuda:
#         mean = mean.cuda()
#         std = std.cuda()
#     image = image.mul(std).add(mean) * 255.0
#     image = image.data.squeeze(0).permute(1, 2, 0).data.cpu().numpy().astype(np.uint8)
#     return image

# def save_plot(dir_name, img_name):
#     my_path = os.path.abspath(os.getcwd() + dir_name)
#     print(my_path)
#     _dir = os.path.join(my_path, img_name)   
#     plt.savefig(_dir, bbox_inches='tight')