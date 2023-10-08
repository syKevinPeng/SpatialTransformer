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
python3 main.py \
--train \
--to_gray \
--epoch 400 \
--bat_size 64 \
--network FlowNetC \
--loss multiscale \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/ckpts/exp9_gray_whole_FlowNetC_multiscale${current_date_time} \
--save_frequency 100 \
--name "exp8 whole gray FlowNetC Continued" \
--notes "FlowNetC (With Correlation) model trained on ChairsSDHom dataset. Multiscale loss" \
--wandb_mode "disabled" # wandb model can be "online", "offline", "disabled" 

# for inference
# python3 main.py \
# --eval \
# --bat_size 64 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp1_viz_evaluation_${current_date_time} \
# --save_frequency 100 \
# --name "exp1 viz - evaluation" \
# --notes "Validate on ChairSDHum Dataset" \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp1_2023-09-07_18:13:23/net_epoch_900.pth" \
# --wandb_mode "disabled" 


# for test
# python3 main.py \
# --test \
# --to_gray \
# --bat_size 64 \
# --network FlowNetSD \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/test_${current_date_time} \
# --save_frequency 100 \
# --name "exp8 viz - normal ouchi" \
# --notes "FlowNetC model trained on ChairsSDHom dataset." \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp8_gray_whole_FlowNetC_2023-10-04_21:21:57/net_epoch_1000.pth" \
# --wandb_mode "disabled" \
# --test_dataset_path ${small_ouchi_0_255}



# python3 main.py \
# --test \
# --bat_size 64 \
# --network FlowNetSD \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp6_small_ouchi_100-140_${current_date_time} \
# --name "exp6 gray whole l1 viz on small 100-140 ouchi" \
# --notes "FlowNetC model trained on ChairsSDHom dataset. " \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp8_gray_whole_FlowNetC_2023-09-29_20:15:10/net_epoch_399.pth" \
# --wandb_mode "online" \
# --test_dataset_path ${small_ouchi_100_140}