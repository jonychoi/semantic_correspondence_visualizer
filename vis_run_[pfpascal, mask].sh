CUDA_VISIBLE_DEVICES=0 python ./vis_run.py \
--dataset 'pfpascal' \
--datapath '../../Datasets_CATs' \
--confmatch_pretrained_path '../model_weights/confmatch_pascal_best.pth' \
--cats_pretrained_path '../model_weights/cats_pascal_best.pth' \
--chm_pretrained_path '' \
--semimatch_pretrained_path '' \
--kps_or_mask 'mask' \
--save_dir './imgs/' \