from visualize_utils.util import flow2kps
from visualize_utils.mask_vis import mask_plotter

def predict_confmatch(i, net, mini_batch, args, device):
    if (args.kps_or_mask == "mask"):
        _, _, _, confidence_map = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device), branch='conf')
        print("#####", confidence_map.shape)
        mask_plotter(args, "confmatch", i, confidence_map, save_dir = args.save_dir+'/confidence_mask/confmatch')

    elif (args.kps_or_mask == "kps"):
        pred_map, _, _, _ = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)
        #estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
        return None

def predict_cat(i, net, mini_batch, args, device):
    if (args.kps_or_mask == "mask"):
        confidence_map = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device), args)
        print("#####", confidence_map.shape)
        mask_plotter(args, "cats", i, confidence_map, args.save_dir+'/confidence_mask/cats')

    elif (args.kps_or_mask == "kps"):
        pred_map, _, _, _ = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)
        #estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
        return None