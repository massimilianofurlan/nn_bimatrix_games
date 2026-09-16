#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --output=out/%j.txt

module purge

module load GCC/11.3.0 Python/3.10.4 texlive/20230313
source .localenv/bin/activate

# evaluate 2x2 default 
srun --unbuffered python -m src.scripts.evaluation.evaluate_learning --model=2x2_default --dataset=2x2_default_avon

# evaluate 3x3 default 
srun --unbuffered python -m src.scripts.evaluation.evaluate_learning --model=3x3_default --dataset=3x3_default_avon

