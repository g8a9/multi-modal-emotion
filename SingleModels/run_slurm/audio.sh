#!/bin/bash

#SBATCH --job-name=audio_single

# Give job a name

#SBATCH --time 00-20:00 # time (DD-HH:MM)

#SBATCH --nodes=1

#SBATCH --gpus-per-node=v100l:1 # request GPU

#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --mem=100G # memory per node

#SBATCH --account=def-whkchun # Runs it on the dedicated nodes we have

#SBATCH --output=/scratch/prsood/audio_single/logs/%N-%j.out # %N for node name, %j for jobID # Remember to mae logs-dir

module load StdEnv/2020

module load cuda

module load cudnn/8.0.3

source /home/prsood/projects/def-whkchun/prsood/sarcasm_venv/bin/activate

# CUDA_VISIBLE_DEVICES=0 to select GPU 0 for our run
# parallel --joblog /scratch/zeerak/MTL/logs/parallel.log < /scratch/zeerak/MTL/configs/sweep_commands.txt

wandb agent ddi/Audio/19mj3gjx --count 20 
# python3 /home/prsood/projects/def-whkchun/prsood/multi-modal-sarcasm/text_nn.py
