#!/usr/bin/bash 
#SBATCH --time=2-00:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err

set -x
echo "----------- INFO ------------"
echo "Train with SpatialTransformer on ChairsSDHom"
echo "Experiment ID: exp2_RGB_1000"
echo "Output Dir: /vulcanscratch/peng2000/SpatialTransformer/exp2_RGB_1000"
echo "-----------------------------"

srun zsh -c "conda activate /vulcanscratch/peng2000/raft; \
python3 train_flownet_sd.py \
--write \
--train \
--resume \
--to_gray False \
--network flownet \
--dataset_path /vulcanscratch/peng2000/ChairsSDHom/data \
--save_path /vulcanscratch/peng2000/SpatialTransformer/exp2_RGB_1000 \
--save_frequency 10 \
--epoch 1200 "


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

