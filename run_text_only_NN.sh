#!/bin/bash

#SBATCH --job-name=text_nn_only

# Give job a name

#SBATCH --time 00-20:00 # time (DD-HH:MM)

#SBATCH --nodes=1

#SBATCH --gpus-per-node=v100l:4 # request GPU

#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --mem=24G # memory per node

#SBATCH --account=ctb-whkchun # Runs it on the dedicated nodes we have

#SBATCH --output=/scratch/prsood/text_nn_only/logs/%N-%j.out # %N for node name, %j for jobID # Remember to mae logs-dir

module load StdEnv/2020

module load cuda

module load cudnn/8.0.3

source /home/prsood/projects/def-whkchun/prsood/sarcasm_venv/bin/activate

wandb agent ddi/at_end_text_bert_NN/z3vjayn6 --count 20
# python3 /home/prsood/projects/def-whkchun/prsood/multi-modal-sarcasm/text_nn.py
