#### Training Baseline Models

**2×2 Baseline**

Train on 2×2 games
```bash
python -m src.scripts.training.train \
  --batch_size=128 \
  --gamma=0.999999 \
  --optimizer=SGD \
  --lr=0.500 \
  --n_games=1073741824 \
  --config=2x2_default \
  --name=2x2_default \
  --log_models \
  --seed=1
```

**3×3 Baseline**

Train on 3×3 games (note: log_models may be demanding):
```bash
python -m src.scripts.training.train \
  --batch_size=128 \
  --gamma=0.999999 \
  --optimizer=SGD \
  --lr=0.500 \
  --n_games=1073741824 \
  --config=3x3_default \
  --name=3x3_default \
  --log_models \
  --seed=1
```

Note: the --log_models flag is only required to reproduce Figures 5 and 6; you can disable it to avoid unnecessary disk writes.


#### Generating evaluation sets  

Generate (labeled) dataset set of 2 x 2 games 
```bash
python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=2 --n_traces=10000 --name=2x2_default
```

Generate (labeled) dataset set of 3 x 3 games
```bash
python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_preferences --game_class=general_sum --n_actions=3 --n_traces=10000 --name=3x3_default
```

Note: reduce the --n_traces value to speed up execution (minimum is 0); higher values yield more accurate equilibrium‐selection results (Table 4)


#### Evaluating the networks 

**2×2 Baseline**

Evaluate on all games
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_default --dataset=2x2_default
```

Evaluate selection
```bash
python -m src.scripts.evaluation.evaluate_selection --model=2x2_default --dataset=2x2_default
```

Evaluate play on affine transformations
```bash
python -m src.scripts.evaluation.evaluate_outsample --model=2x2_default --dataset=2x2_default
```

Evaluate abidance to axioms
```bash
python -m src.scripts.evaluation.evaluate_axioms --model=2x2_default --dataset=2x2_default
```

Evaluate speed of learning (2x2_default must be trained using --log_models)
```bash
python -m src.scripts.evaluation.evaluate_learning --model=2x2_default --dataset=2x2_default
```

Evaluate play on strategic subspace 
```bash
python -m src.scripts.evaluation.evaluate_2x2_strategic --model=2x2_default
```

**3×3 Baseline**

Evaluate on all games
```bash
python -m src.scripts.evaluation.evaluate --model=3x3_default --dataset=3x3_default
```

Evaluate selection
```bash
python -m src.scripts.evaluation.evaluate_selection --model=3x3_default --dataset=3x3_default
```

Evaluate play on affine transformations
```bash
python -m src.scripts.evaluation.evaluate_outsample --model=3x3_default --dataset=3x3_default
```

Evaluate abidance to axioms
```bash
python -m src.scripts.evaluation.evaluate_axioms --model=3x3_default --dataset=3x3_default
```

Evaluate speed of learning 
```bash
python -m src.scripts.evaluation.evaluate_learning --model=3x3_default --dataset=3x3_default
```



#### Robustness — Subspaces

**2×2 subspace tests**

Train on three 2×2 subspaces and non-uniform payoff sampling:

```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 \
  --config=2x2_subspace1 --name=2x2_subspace1 --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 \
  --config=2x2_subspace2 --name=2x2_subspace2 --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 \
  --config=2x2_subspace3 --name=2x2_subspace3 --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 \
  --config=2x2_nonuniform --name=2x2_nonuniform --seed=1
```

Evaluate play all games
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_subspace1 --dataset=2x2_default
python -m src.scripts.evaluation.evaluate --model=2x2_subspace2 --dataset=2x2_default
python -m src.scripts.evaluation.evaluate --model=2x2_subspace3 --dataset=2x2_default
```



#### Robustness — Loss Functions

**2×2 loss variants**

Train by minimizing linear regret, ex-post regret, and squared ex-post regret:

```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.005 --n_games=1073741824 \
  --config=2x2_linear_loss --name=2x2_linear_loss --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.005 --n_games=1073741824 \
  --config=2x2_expost_loss --name=2x2_expost_loss --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.005 --n_games=1073741824 \
  --config=2x2_expost_loss_sq --name=2x2_expost_loss_sq --seed=1
```



#### Robustness — Hyperparameters

**2×2 hyperparameter tests**

Train with batch size=1 (online), half and double batch sizes, and with no learning rate decay:

```bash
python -m src.scripts.training.train --batch_size=1 --gamma=0.999999 \
  --optimizer=SGD --lr=0.005 --n_games=8388608 --config=2x2_default \
  --name=2x2_nobatch --seed=1
python -m src.scripts.training.train --batch_size=64 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=536870912 --config=2x2_default \
  --name=2x2_halfbatch --seed=1
python -m src.scripts.training.train --batch_size=256 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=2147483648 --config=2x2_default \
  --name=2x2_doublebatch --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=1 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_default \
  --name=2x2_nolrdecay --seed=1
```



#### Robustness — Architecture

Train smaller, bigger and asymmetric networks:

```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_halved_neurons \
  --name=2x2_halved_neurons --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_doubled_neurons \
  --name=2x2_doubled_neurons --seed=1
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=1073741824 --config=2x2_asymmetric \
  --name=2x2_asymmetric --seed=1
```



#### Robustness — Larger Games

Train networks to play 4×4 and 5×5 games:

```bash
python -m src.scripts.training.train --batch_size=512 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=4294967296 --config=4x4_default \
  --name=4x4_default --seed=1
python -m src.scripts.training.train --batch_size=512 --gamma=0.999999 \
  --optimizer=SGD --lr=0.500 --n_games=4294967296 --config=5x5_default \
  --name=5x5_default --seed=1
```
