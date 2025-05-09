#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --output=out/%j.txt

module purge

#module load CUDA/11.7.0
#module load GCCcore/12.2.0 Python/3.10.8
#module load GCC/12.2.0 OpenMPI/4.1.4 PyTorch/2.1.2 SciPy-bundle/2023.02

#source .localenv/bin/activate

module load GCC/11.3.0
module load OpenMPI/4.1.4
module load SciPy-bundle/2022.05
module load matplotlib/3.5.2
module load PyTorch/1.12.1-CUDA-11.7.0
module load CUDA/11.7.0
module load GCCcore/11.3.0
module load Python/3.10.4
module load texlive/20230313

srun --unbuffered python3.10 -m src.scripts.evaluation.evaluate --model=$1 --dataset=$2
