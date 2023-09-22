#!/bin/bash
current_date_time=`date +"%Y-%m-%d_%T"`
ouchi_path = "/home/siyuan/research/SpatialTransformer/ckpts/exp6_gray_whole_l1_2023-09-21_14:38:47/net_epoch_199.pth"
#for training
# python3 main.py \
# --train \
# --epoch 200 \
# --bat_size 64 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp6_gray_whole_l1_${current_date_time} \
# --save_frequency 100 \
# --name "exp6 whole gray Change of Loss Term" \
# --notes "flownet model trained on ChairsSDHom dataset. change of gradient_loss coefficient to 0.005 from 0.1" \
# --wandb_mode "online" # wandb model can be "online", "offline", "disabled"

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
python3 main.py \
--test \
--bat_size 64 \
--network flownet \
--loss unsup_loss \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/ckpts/exp6_viz_ouchi_${current_date_time} \
--save_frequency 100 \
--name "exp6 viz - ouchi" \
--notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with all gray images. visulize on normal ouchi. Note: less smoothness applied" \
--load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp6_gray_whole_l1_2023-09-21_14:38:47/net_epoch_199.pth" \
--wandb_mode "online" \
--test_dataset_path "/home/siyuan/research/SpatialTransformer/data/brick_FLOW"



# python3 main.py \
# --test \
# --to_gray \
# --bat_size 64 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp3_texture4_${current_date_time} \
# --save_frequency 100 \
# --name "exp3 gray whole texture 4" \
# --notes "flownet model trained on ChairsSDHom. Visulize texture4, move upward" \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp3_Gray_whole/net_epoch_999.pth" \
# --wandb_mode "online" \
# --test_dataset_path "/home/siyuan/research/SpatialTransformer/data/texture4_FLOW"