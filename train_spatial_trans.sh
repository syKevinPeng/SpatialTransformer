# for training
# python3 train_flownet_sd.py \
# --write \
# --train \
# --bat_size 32 \
# --resume \
# --to_gray False \
# --network flownet \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole \
# --save_frequency 10 \
# --epoch 1000 


# for inference
python3 train_flownet_sd.py \
--eval \
--bat_size 32 \
--to_gray False \
--network flownet \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole/viz \
--name "Viz RGB whole" \
--notes "Vizsulization of the RGB image on the whole dataset" \
--load_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole/net_epoch_999.pth \
--write

