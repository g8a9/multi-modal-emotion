#!/bin/bash

#SBATCH --job-name=tav_mae

# Give job a name

#SBATCH --time 03-12:00 # time (DD-HH:MM)

#SBATCH --nodes=1

#SBATCH --gpus-per-node=v100l:2 # request GPU

#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --mem=180G # memory per node

#SBATCH --account=def-whkchun # Runs it on the dedicated nodes we have

#SBATCH --output=/scratch/prsood/tav_mae/logs/%N-%j.out # %N for node name, %j for jobID # Remember to mae logs-dir

module load StdEnv/2020

module load cuda

module load cudnn/8.0.3

source /home/prsood/projects/def-whkchun/prsood/sarcasm_venv/bin/activate
parallel --joblog /scratch/prsood/tav_mae/logs/parallel.log < ./meld_multi.txt
# wandb agent ddi/MAEncoder_7Emo_Test/v88xjdsy
