#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=3700
#SBATCH --gres=gpu:quadro_rtx_6000:3
#SBATCH --output=out/%j.txt

module purge

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

# test
srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=64 --gamma=0.999995 --optimizer=SGD --lr=0.5 --n_games=536870912 --config=2x2_example --seed=1


# 2x2 baseline (w/ --log_models)
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_default --log_models --seed=1

# 3x3 baseline (w/ --log_models)
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=3x3_default --name=3x3_default --log_models --seed=1


# robustness - subspaces
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_subspace1 --name=2x2_subspace1 --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_subspace2 --name=2x2_subspace2 --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_subspace3 --name=2x2_subspace3 --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_nonuniform --name=2x2_nonuniform --seed=1

# robustness - loss function
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.005 --n_games=1073741824 --config=2x2_linear_loss --name=2x2_linear_loss --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.005 --n_games=1073741824 --config=2x2_expost_loss --name=2x2_expost_loss --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.005 --n_games=1073741824 --config=2x2_expost_loss_sq --name=2x2_expost_loss_sq --seed=1

# robustness - hyperparameters
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=1   --gamma=0.999999 --optimizer=SGD --lr=0.005 --n_games=8388608    --config=2x2_default --name=2x2_nobatch --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=64  --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=536870912  --config=2x2_default --name=2x2_halfbatch --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=256 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=2147483648 --config=2x2_default --name=2x2_doublebatch --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=1        --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_nolrdecay --seed=1

# robustness - architecture
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_halved_neurons --name=2x2_halved_neurons --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_doubled_neurons --name=2x2_doubled_neurons --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_asymmetric --name=2x2_asymmetric --seed=1

# robustness - randomness
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed2 --seed=2
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed3 --seed=3
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed5 --seed=5
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed7 --seed=7
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed11 --seed=11
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed13 --seed=13
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed17 --seed=17
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed19 --seed=19
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed23 --seed=23
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default --name=2x2_seed29 --seed=29

# robustness - larger games 
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=512 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=4294967296 --config=4x4_default --name=4x4_default --seed=1
#srun --unbuffered python3.10 -m src.scripts.training.train --batch_size=512 --gamma=0.999999 --optimizer=SGD --lr=0.500 --n_games=4294967296 --config=5x5_default --name=5x5_default --seed=1
