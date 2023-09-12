#!/bin/bash
current_date_time=`date +"%Y-%m-%d_%T"`

#for training
# python3 main.py \
# --train \
# --epoch 1000 \
# --bat_size 64 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp1_${current_date_time} \
# --save_frequency 100 \
# --name "RGB Flownet all images" \
# --notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with all images" \
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
--to_gray \
--bat_size 64 \
--network flownet \
--loss unsup_loss \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/ckpts/exp2_viz_horizontal_${current_date_time} \
--save_frequency 100 \
--name "exp2 viz - horizontal" \
--notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with 1000 gray-scale images. Visulize on Ouchi image, horizontal pattern only" \
--load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp2_Gray_1000/net_epoch_970.pth" \
--wandb_mode "disabled" \
--test_dataset_path "/home/siyuan/research/SpatialTransformer/data/horizontal_ouchi_FLOW"



# python3 main.py \
# --test \
# --bat_size 64 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/test_${current_date_time} \
# --save_frequency 100 \
# --name "exp1 viz - horizontal" \
# --notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with all RGB images. Visulize on Ouchi image, horizontal pattern only" \
# --load_path "/home/siyuan/research/SpatialTransformer/ckpts/exp1_2023-09-07_18:13:23/net_epoch_900.pth" \
# --wandb_mode "disabled" \
# --test_dataset_path "/home/siyuan/research/SpatialTransformer/data/horizontal_ouchi_FLOW"