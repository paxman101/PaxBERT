#!/bin/bash -l
#SBATCH -J HorovodTFGPU
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 4
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH -p gpus
#SBATCH -c 12
#SBATCH --mem-per-gpu=62G

export MODULEPATH=$MODULEPATH:/home/pschweigert/.modules

conda activate nlp_tf
module load cuda-10.0
module load 20.11

## Horovod execution
horovodrun -np $SLURM_NTASKS -H gpu01:2,gpu02:2,gpu03:2,gpu04:2 python nlp_horovod.py
