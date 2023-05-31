#!/usr/bin/bash 
#SBATCH --time=0-24:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err

set -x
echo "----------- INFO ------------"
echo "Train with SpatialTransformer on ChairsSDHom"
echo "Experiment ID: exp3_gray"
echo "Output Dir: /vulcanscratch/peng2000/SpatialTransformer/exp3_gray"
echo "-----------------------------"

srun zsh -c "conda activate /vulcanscratch/peng2000/raft; python3 train_flownet_sd.py --write --train --to_gray True --network flownet --dataset_path /vulcanscratch/peng2000/ChairsSDHom/data --save_path /vulcanscratch/peng2000/SpatialTransformer/exp3_gray --epoch 100 "

