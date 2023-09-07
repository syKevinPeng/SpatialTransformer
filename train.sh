#for training
python3 main.py \
--train \
--epoch 1000 \
--bat_size 32 \
--network flownet \
--loss unsup_loss \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/ckpts/exp0 \
--save_frequency 100 \
--name "RGB Flownet 1000 images" \
--notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with only 1000 RGB images" \
--num_img_to_train 1000 \
--wandb_mode "disabled" # wandb model can be "online", "offline", "disabled"

# for inference
# python3 train_flownet_sd.py \
# --eval \
# --bat_size 32 \
# --to_gray False \
# --network flownet \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/my_results/unsup_flownet_gray_whole \
# --name "Eval whole gray unsupervised" \
# --notes "Evalation of model's performance with unsupervised loss" \
# --load_path /home/siyuan/research/SpatialTransformer/my_results/unsup_flownet_gray_whole/net_epoch_1100.pth \
# --write

#/home/siyuan/research/SpatialTransformer/my_results/biased_flow_gray_1000/exp1_gray_1000/net_epoch_970.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_gray_whole/net_epoch_999.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_1000/exp2_RGB_1000/net_epoch_1000.pth
# /home/siyuan/research/SpatialTransformer/my_results/biased_flow_rgb_whole/net_epoch_999.pth

# for test
# python3 train_flownet_sd.py \
# --test \
# --to_gray \
# --bat_size 32 \
# --network flownet \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/ckpts/exp0 \
# --save_frequency 100 \
# --name "RGB Flownet 1000 images" \
# --notes "Flownet model (with no correlation) trained on ChairsSDHom dataset with only 1000 images" \