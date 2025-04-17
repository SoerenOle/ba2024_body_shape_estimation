#!/bin/bash
#SBATCH --array=0-31
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
LEARNING_RATES=(0.0001 0.0003 0.0005 0.0007 0.0009 0.001 0.003 0.005 0.007 0.009 0.01 0.03 0.05 0.07 0.09 0.1 0.3 0.5 0.7 0.9 1 3 5 7 9 10 30 50 70 90 0.2 0.4)
/home/g037552/.conda/envs/BA/bin/python /home/g037552/Bachelorarbeit/ba/main.py $@ --lr=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
