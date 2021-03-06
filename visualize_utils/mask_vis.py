import torch
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import os
from visualize_utils.tgt_test_imgs import pfpascal

def mask_plotter(args, model_name, index, confidence_map, test_anno, theme = plt.cm.hot, upsample_mode = 'bilinear', name_plot = False, name = None):
    confidence_map = upsampling(args, confidence_map, mode = upsample_mode).to('cpu')

    #print(name, confidence_map.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #resizer
    
    if name_plot:
        plt.title(name, size=14)

    if args.dataset == 'pfpascal':
        image = Image.open('/media/ssd/Datasets_CATs/PF-PASCAL/'+pfpascal()[index])
        if args.first_masking_order == "image":
            dir = args.save_dir+'confidence_mask/{}/{}/order_inverted/image opacity of- {}/threshold of- {}'.format(args.dataset, model_name, args.image_opacity, args.threshold)
        else:
            dir = args.save_dir+'confidence_mask/{}/{}/image opacity of- {}/threshold of- {}'.format(args.dataset, model_name, args.image_opacity, args.threshold)
        
    elif args.dataset == 'spair':
        anno = test_anno[index].split(":")
        label = anno[1]
        tgt = anno[0].split("-")[2]
        image = Image.open('/media/ssd/Datasets_CATs/SPair-71k/JPEGImages/{}/{}.jpg'.format(label, tgt))
        if args.first_masking_order == "image":
            dir = args.save_dir+'confidence_mask/{}/{}/order_inverted/image opacity of- {}/threshold of- {}/{}'.format(args.dataset, model_name, args.image_opacity, args.threshold, label)
        else:
            dir = args.save_dir+'confidence_mask/{}/{}/image opacity of- {}/threshold of- {}/{}'.format(args.dataset, model_name, args.image_opacity, args.threshold, label)


    resized_image = image.resize((256, 256)) # Use PIL to resize
    plt.axis('off')

    if args.first_masking_order == 'image':
        a = ax.imshow(resized_image, alpha=args.image_opacity)

        map = ax.imshow(confidence_map, cmap=theme, alpha=1)
    else: #default => mask first and image front
        map = ax.imshow(confidence_map, cmap=theme, alpha=1)
    
        a = ax.imshow(resized_image, alpha=args.image_opacity)

    cb = fig.colorbar(map, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    #cb.ax.text(1.88, 1, 'Confidence', va='bottom', ha='center') #r'$\times$10$^{-1}$'
    
    isExist = os.path.exists(dir)
    if not isExist:    
        os.makedirs(os.getcwd()+dir, exist_ok = True)

    save_plot(dir_name=dir, img_name="{}'s {} confidence map of {}".format(args.dataset, index, model_name))
    plt.close()
    return image.close()

def upsampling(args, confidence_map, scale_factor = 16, mode = 'bilinear'):
    #confidence_map = confidence_map.unsqueeze(0).unsqueeze(0) # make 4D from 2D
    if confidence_map.dim() == 3:
        confidence_map = confidence_map.unsqueeze(0)
    upsampler = nn.Upsample(scale_factor = scale_factor, mode = mode)
    upsampled_confidence_map = upsampler(confidence_map)
    _2d_map = upsampled_confidence_map.squeeze(0).squeeze(0)

    zeros = torch.zeros_like(_2d_map)

    # if args.threshold:
    #     _2d_map = torch.where(_2d_map >= args.threshold, _2d_map, zeros)
    #print(args.threshold, len(_2d_map >= args.threshold))
    #print(_2d_map)
    _2d_map = torch.where(_2d_map >= args.threshold, _2d_map, zeros)
    return _2d_map
    
def save_plot(dir_name, img_name):
    my_path = os.path.abspath(os.getcwd() + dir_name)
    #print(my_path)
    _dir = os.path.join(my_path, img_name)   
    plt.savefig(_dir, bbox_inches='tight', dpi = 100)
    
dummy = torch.FloatTensor([[0.12, 0.1, 0.12, 0.2, 0.23, 0.125, 0.128, 0.1, 0.122, 0.22, 0.25, 0.1, 0.23, 0.122, 0.22, 0.125],
                           [0.12, 0.12, 0.12, 0.122, 0.23, 0.125, 0.1, 0.1, 0.122, 0.22, 0.25, 0.1, 0.23, 0.122, 0.22, 0.125],
                           [0.12, 0.1, 0.12, 0.122, 0.23, 0.125, 0.1, 0.128, 0.122, 0.22, 0.1, 0.128, 0.23, 0.122, 0.22, 0.125],
                           [0.12, 0.12, 0.11, 0.122, 0.23, 0.125, 0.128, 0.128, 0.122, 0.22, 0.1, 0.128, 0.23, 0.122, 0.22, 0.125],
                           [0.12, 0.12, 0.1, 0.122, 0.23, 0.125, 0.128, 0.128, 0.122, 0.22, 0.1, 0.56, 0.23, 0.122, 0.22, 0.125],
                           [0.12, 0.12, 0.12, 0.122, 0.23, 0.125, 0.128, 0.128, 0.122, 0.99, 0.76, 0.128, 0.55, 0.122, 0.22, 0.125],
                           [0.21, 0.12, 0.12, 0.122, 0.55, 0.99, 0.55, 0.99, 0.122, 0.99, 0.76, 0.128, 0.55, 0.122, 0.88, 0.125],
                           [0.01, 0.12, 0.12, 0.122, 0.55, 0.125, 0.99, 0.128, 0.122, 0.88, 0.25, 0.128, 0.55, 0.122, 0.88, 0.125],
                           [0.12, 0.12, 0.54, 0.88, 0.55, 0.125, 0.128, 0.97, 0.122, 0.88, 0.25, 0.128, 0.55, 0.122, 0.22, 0.125],
                           [0.512, 0.12, 0.12, 0.122, 0.55, 0.125, 0.128, 0.128, 0.122, 0.22, 0.89, 0.128, 0.55, 0.19, 0.22, 0.35],
                           [0.3, 0.3, 0.3, 0.19, 0.55, 0.35, 0.38, 0.38, 0.88, 0.22, 0.25, 0.38, 0.55, 0.19, 0.22, 0.35],
                           [0.3, 0.3, 0.3, 0.19, 0.55, 0.35, 0.38, 0.38, 0.19, 0.22, 0.25, 0.38, 0.55, 0.19, 0.22, 0.35],
                           [0.78, 0.3, 0.3, 0.19, 0.35, 0.57, 0.38, 0.38, 0.19, 0.22, 0.25, 0.18, 0.55, 0.19, 0.22, 0.35],
                           [0.3, 0.3, 0.3, 0.19, 0.05, 0.15, 0.38, 0.1, 0.19, 0.22, 0.25, 0.1, 0.18, 0.19, 0.22, 0.35],
                           [0.3, 0.3, 0.3, 0.19, 0.08, 0.25, 0.1, 0.1, 0.19, 0.22, 0.25, 0.38, 0.18, 0.285, 0.22, 0.35],
                           [0.3, 0.3, 0.3, 0.21, 0.06, 0.25, 0.38, 0.38, 0.555, 0.22, 0.25, 0.38, 0.25, 0.15, 0.22, 0.35]])

#mask_plotter('./demo_imgs/2007_003715.jpg', dummy, save_dir = './confidence_masks_imgs', plt.cm.hot, upsample_mode = 'bilinear', name="Driven Confidence of ConfMatch_winter")
#mask_plotter('./demo_imgs/2007_003711.jpg', dummy, save_dir = './confidence_masks_imgs', plt.cm.hot, upsample_mode = 'nearest', name="Not Upsampled Confidence")