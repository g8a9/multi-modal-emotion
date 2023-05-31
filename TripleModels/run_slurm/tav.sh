#!/bin/bash

#SBATCH --job-name=TAV_MAE

# Give job a name

#SBATCH --time 00-00:20 # time (DD-HH:MM)

#SBATCH --nodes=1

#SBATCH --gpus-per-node=v100l:1 # request GPU

#SBATCH --ntasks-per-node=4

#SBATCH --cpus-per-task=6 # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.

#SBATCH --mem=50G # memory per node

#SBATCH --account=def-whkchun # Runs it on the dedicated nodes we have

#SBATCH --output=/scratch/prsood/tav_mae/logs/%N-%j.out # %N for node name, %j for jobID # Remember to mae logs-dir

module load StdEnv/2020

module load gcc/9.3.0  

module load cuda/11.4

module load python/3.8


# module load cudnn/8.0.3

# module load opencv/4.7.0

source /home/prsood/projects/def-whkchun/prsood/38Venv/bin/activate

# wandb agent ddi/MAEncoder_7Emo_Test/v88xjdsy --count 20
# parallel --joblog /scratch/prsood/tav_mae/logs/parallel.log < ./meld_multi.txt
# wandb agent ddi/MAEncoder_7Emo_Test/v88xjdsy
# wandb agent ddi/MAEncoder_7Emo_Test/q6kuaixc #This is the new one!!!!
wandb agent ddi/MAEncoder_7Emo_Test/z7kzs6go
