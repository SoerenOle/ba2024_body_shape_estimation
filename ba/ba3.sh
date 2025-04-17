#!/bin/bash
#SBATCH --array=0-9
#SBATCH --job-name="Body-Outlines"
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=0-23:59:59
#SBATCH --output=logs/%j.out
#SBATCH --error=runs/%j.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --mail-user=soeren.kottner@student.uni-siegen.de
#SBATCH --chdir=/home/g037552/Projects/Bachelorarbeit
#SBATCH --output=/home/g037552/output_msg_sbatch%j.txt
#SBATCH --error=/home/g037552/errror_msg_sbatch%j.txt
source /cm/shared/omni/apps/miniconda3/bin/activate BA
LEARNING_RATES=(0.00001 0.00003 0.00005 0.000009 0.000007 0.000005 0.0001 0.0003 0.0005 0.0007)
/home/g037552/.conda/envs/BA/bin/python /home/g037552/Bachelorarbeit/ba/main.py $@ --lr=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}


