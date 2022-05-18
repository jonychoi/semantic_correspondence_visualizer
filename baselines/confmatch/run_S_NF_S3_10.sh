CUDA_VISIBLE_DEVICES=0,1,2,3 python train_confmatch_ETE.py --batch-size 64 \
--lr_con 0.000001 \
--freeze False --conf_est_resume True \
--start_epoch 11 --pretrained '../conf_pretrained_models/model_best_35_0.73.pth' \
--keyout 0.2 \
--contrastive_gt_mask --refined_corr_filtering 'mutual' --additional_weak \
--step '[80, 100, 110]' --step_con '[80, 100, 110]' \
--benchmark spair --hyperpixel '[0,8,20,21,26,28,29,30]' --dynamic_unsup \
--data_parallel \
--use_maxProb False \
--con_est_type 'tf_not_residual_shallow' \
--con_est_depth 1 \
--alpha_train 0.1 \
--alpha_train_self 0.1 \
--cut_off 0.9 \
--con_est_input_map_type unNorm \
--name_exp 'Final_S_10_self01_NF_015_confmatch_cutoff_09_self01' \
--vis_interval 1000 
