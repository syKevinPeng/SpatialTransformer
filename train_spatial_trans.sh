python3 train_flownet_sd.py \
--write \
--train \
--to_gray False \
--network flownet \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole \
--save_frequency 100 \
--epoch 1000 \ 
--bat_size 64


# python3 train_flownet_sd.py \
# --write \
# --eval \
# --resume \
# --to_gray False \
# --network flownet \
# --dataset_path /vulcanscratch/peng2000/ChairsSDHom/data \
# --save_path ./my_results/biased_flow_rgb_1000 \
# --save_frequency 10 \
# --epoch 1200

