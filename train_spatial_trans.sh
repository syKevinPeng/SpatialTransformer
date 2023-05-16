#!/usr/bin/bash 
#SBATCH --time=1-24:00:00
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_output/slurm-%j.out
#SBATCH --error=slurm_output/slurm-%j.err

set -x
cd Optical_Illusion
cd SpatialTransformer
echo "----------- INFO ------------"
echo "Train with SpatialTransformer on ChairsSDHom"
echo "Experiment ID: exp0"
echo "Output Dir: /vulcanscratch/peng2000/SpatialTransformer/exp0"
echo "-----------------------------"

srun zsh -c "conda activate raft; \
    python3 train_flownet_sd.py -w \
                            --dataset_path /vulcanscratch/peng2000/ChairsSDHom/data \
                            --save_path /vulcanscratch/peng2000/SpatialTransformer/exp0 \"