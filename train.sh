#!/bin/bash
current_date_time=`date +"%Y-%m-%d_%T"`
ouchi_0_255_path="/home/siyuan/research/SpatialTransformer/data/ouchi_0-255_FLOW"
ouchi_40_200_path="/home/siyuan/research/SpatialTransformer/data/ouchi_40-200_FLOW"
ouchi_60_180_path="/home/siyuan/research/SpatialTransformer/data/ouchi_60-180_FLOW"
ouchi_80_160_path="/home/siyuan/research/SpatialTransformer/data/ouchi_80-160_FLOW"
small_ouchi_0_255="/home/siyuan/research/SpatialTransformer/data/small_ouchi_0-255_FLOW"
small_ouchi_40_200="/home/siyuan/research/SpatialTransformer/data/small_ouchi_40-200_FLOW"
small_ouchi_60_180="/home/siyuan/research/SpatialTransformer/data/small_ouchi_60-180_FLOW"
small_ouchi_80_160="/home/siyuan/research/SpatialTransformer/data/small_ouchi_80-160_FLOW"
small_ouchi_100_140="/home/siyuan/research/SpatialTransformer/data/small_ouchi_100-140_FLOW"


#for training
# python3 main.py \
# --train \
# --to_gray \
# --epoch 800 \
# --bat_size 64 \
# --resume \
# --network FlowNetC \
# --loss multiscale \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp9_gray_whole_FlowNetC_multiscale_${current_date_time} \
# --save_frequency 100 \
# --name "exp9 whole gray FlowNetC with Multiscale loss" \
# --load_path /home/siyuan/research/SpatialTransformer/ckpts/exp9_gray_whole_FlowNetC_multiscale_2023-10-07_22:28:04/net_epoch_399.pth \
# --notes "FlowNetC (With Correlation) model trained on ChairsSDHom dataset. Multiscale loss" \
# --wandb_mode "online" # wandb model can be "online", "offline", "disabled" 

# for inference
# python3 main.py \
# --eval \
# --bat_size 64 \
# --network FlowNetSD \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp6_inference_${current_date_time} \
# --save_frequency 100 \
# --name "exp6 viz - evaluation" \
# --notes "Validate on ChairSDHum Dataset" \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp6_gray_whole_less_smooth_2023-09-21_14:38:47/net_epoch_199.pth" \
# --wandb_mode "online" 


# for test
python3 main.py \
--test \
--bat_size 64 \
--network FlowNetSD \
--loss unsup_loss \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/ckpts/exp6_viz_texture_7_${current_date_time} \
--save_frequency 100 \
--name "exp6 viz - texture 7" \
--notes "FlowNetSD model trained on ChairsSDHom dataset with unsupervised loss." \
--load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp6_gray_whole_less_smooth_2023-09-21_14:38:47/net_epoch_199.pth" \
--wandb_mode "online" \
--test_dataset_path "/home/siyuan/research/SpatialTransformer/data/texture7_FLOW"



# python3 main.py \
# --test \
# --bat_size 64 \
# --to_gray \
# --network FlowNetC \
# --loss multiscale \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp9_small_ouchi_0-255_${current_date_time} \
# --name "exp9 gray whole normal ouchi 600 epochs" \
# --notes "FlowNetC model trained on ChairsSDHom dataset. " \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp9_gray_whole_FlowNetC_multiscale_2023-10-09_19:14:30/net_epoch_600.pth" \
# --wandb_mode "online" \
# --test_dataset_path ${small_ouchi_0_255}