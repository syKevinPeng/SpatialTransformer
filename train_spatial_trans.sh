#for training
python3 train_flownet_sd.py \
--write \
--train \
--bat_size 32 \
--to_gray False \
--network cnn \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/my_results/unsup_cnn_rgb_whole \
--save_frequency 100 \
--name "cnn whole rgb" \
--epoch 1000 \
--notes "Training of the CNN model on the whole dataset of ChairSD unsupervised loss" \


# for inference
# python3 train_flownet_sd.py \
# --eval \
# --bat_size 32 \
# --to_gray False \
# --network flownet \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_1000/ \
# --name "Eval RGB 1000" \
# --notes "Evalation of the model's performance on validation set of ChairSD" \
# --load_path /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_1000/exp2_RGB_1000/net_epoch_1000.pth \
# --write

#/home/siyuan/research/SpatialTransformer/my_results/biased_flow_gray_1000/exp1_gray_1000/net_epoch_970.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_gray_whole/net_epoch_999.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_1000/exp2_RGB_1000/net_epoch_1000.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole/net_epoch_999.pth

# for test
# python3 train_flownet_sd.py \
# --test \
# --bat_size 32 \
# --to_gray False \
# --network cnn \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/my_results/unsup_cnn_rgb_whole \
# --name "cnn whole rgb viz" \
# --notes "Visualization of the CNN model on the whole dataset of ChairSD unsupervised learning" \
# --load_path /home/siyuan/research/SpatialTransformer/my_results/unsup_cnn_rgb_whole/net_epoch_999.pth \
# --write
