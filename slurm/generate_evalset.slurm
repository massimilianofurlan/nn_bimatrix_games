#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --output=out/%j.txt

module purge

#module load CUDA/11.7.0
module load GCCcore/12.2.0 Python/3.10.8
module load GCC/12.2.0 OpenMPI/4.1.4 PyTorch/2.1.2 SciPy-bundle/2023.02

source .localenv/bin/activate

# generate evaluation set 2x2 games
srun --unbuffered python3.10 -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=2 --n_traces=10000

# generate evaluation set 3x3 games
srun --unbuffered python3.10 -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=3 --n_traces=10000
