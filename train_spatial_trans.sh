#for training
python3 train_flownet_sd.py \
--train \
--bat_size 128 \
--network pwc \
--dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
--save_path /home/siyuan/research/SpatialTransformer/my_results/unsup_pwc_rgb_whole \
--save_frequency 100 \
--name "pwc whole rgb" \
--epoch 1000 \
--notes "Training of the pwc model on the whole dataset of ChairSD unsupervised loss" \
--loss "unsup_loss" \
--write 


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
# --bat_size 32 \
# --to_gray False \
# --network pwc \
# --loss unsup_loss \
# --dataset_path /home/siyuan/research/dataset/ChairsSDHom/data \
# --save_path /home/siyuan/research/SpatialTransformer/my_results/unsup_flownet_gray_whole_viz \
# --save_frequency 100 \
# --name "test whole RGB pwcnet unsupervised" \
# --notes "generate predictions with unsupervised loss with PWC net" \
# --load_path /home/siyuan/research/PWC-Net-2/PyTorch/pwc_net.pth.tar \