#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --output=out/%j.txt

module purge

module load GCC/11.3.0 Python/3.10.4 texlive/20230313
source .localenv/bin/activate

srun --unbuffered python -m src.scripts.evaluation.evaluate --model=$1 --dataset=$2
