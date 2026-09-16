#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --output=out/%j.txt

module purge

module load GCC/11.3.0 Python/3.10.4
source .localenv/bin/activate

# generate evaluation set 2x2 games
srun --unbuffered python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=2 --n_traces=10000

# generate evaluation set 3x3 games
srun --unbuffered python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=3 --n_traces=10000
