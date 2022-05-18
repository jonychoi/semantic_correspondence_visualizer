from visualize_utils import flow2kps

def predict_confmatch(net, mini_batch, args, device):
    if (args.kps_or_mask == "mask"):
        _, _, _, confidence_map = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        return confidence_map

    elif (args.kps_or_mask == "kps"):
        pred_map, _, _, _ = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)
        estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
        return None

def predict_cat(net, mini_batch, args, device):
    if (args.kps_or_mask == "mask"):
        confidence_map = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        return confidence_map

    elif (args.kps_or_mask == "kps"):
        pred_map, _, _, _ = net(mini_batch['trg_img'].to(device), mini_batch['src_img'].to(device))
        pred_flow = net.unnormalise_and_convert_mapping_to_flow(pred_map)
        estimated_kps = flow2kps(mini_batch['trg_kps'].to(device), pred_flow, mini_batch['n_pts'].to(device))
        return None