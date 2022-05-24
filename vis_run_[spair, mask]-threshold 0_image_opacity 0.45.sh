CUDA_VISIBLE_DEVICES=0 python vis_run.py \
--dataset 'spair' \
--datapath '../../Datasets_CATs' \
--kps_or_mask 'mask' \
--save_dir '/imgs/' \
--seed 1998 \
--threshold 0.0 \
--image_opacity 0.45 \
--confmatch_pretrained_path '../model_weights/confmatch_spair_best.pth'