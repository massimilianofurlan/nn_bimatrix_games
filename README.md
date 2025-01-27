# Code for D. Condorelli and M. Furlan, Deep Learning to Play Games

## Replication instructions

This is a work in progress. Replication instructions might be broken. 
 

#### Clone repository
Clone the repository to your local machine using the following command:
```bash
git clone https://github.com/massimilianofurlan/nn_bimatrix_games.git
```


#### Python environment

Install Python 3.12 (this is what I am using, other versions may work)
```bash
brew install python3.12
```

Set up virtual environment 
```bash
cd nn_bimatrix_games
python3.12 -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

#### Training

Train pair of newtorks on 2 x 2 games (might want to set --log_models=true)
```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_default --name=2x2_default
```

Train pair of newtorks on 3 x 3 games (might want to set --log_models=true)
```bash
python -m src.scripts.training.train --batch_size=256 --gamma=0.9999995 --optimizer=SGD --lr=1 --n_games=2147483648 --config=3x3_default --name=3x3_default
```


#### Generating evaluation sets 

Generate (labeled) dataset set of 2 x 2 games
```bash
python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_orthogonal --game_class=general_sum --n_actions=2 --n_traces=10000 --name=2x2_default
```

Generate (labeled) dataset set of 3 x 3 games
```bash
python -m src.scripts.data.generate_evalset --n_games=131072 --payoffs_space=sphere_orthogonal --game_class=general_sum --n_actions=3 --n_traces=10000 --name=3x3_default
```


#### Generic evaluation

Evaluate on 2 x 2 games
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_default --dataset=2x2_default
```

Evaluate on 3 x 3 games
```bash
python -m src.scripts.evaluation.evaluate --model=3x3_default --dataset=3x3_default
```


#### Evaluating selection

Evaluating selection on 2 x 2 games
```bash
python -m src.scripts.evaluation.evaluate_selection --model=2x2_default --dataset=2x2_default
```

Evaluating selection on 3 x 3 games
```bash
python -m src.scripts.evaluation.evaluate_selection --model=3x3_default --dataset=3x3_default
```


#### Evaluating axioms

Evaluating axioms on 2 x 2 games
```bash
python -m src.scripts.evaluation.evaluate_axioms --model=2x2_default --dataset=2x2_default
```

Evaluating axioms on 3 x 3 games
```bash
python -m src.scripts.evaluation.evaluate_axioms --model=3x3_default --dataset=3x3_default
```


#### Plot learning curves (requires models trained with --log_models=true)

Evaluating learning on 2 x 2 games
```bash
python -m src.scripts.evaluation.evaluate_learning --model=2x2_default --dataset=2x2_default
```

Evaluating learning on 3 x 3 games
```bash
python -m src.scripts.evaluation.evaluate_learning --model=3x3_default --dataset=3x3_default
```


#### Robustness (larger models)

Train pair of newtorks on 4 x 4 games (might want to set --log_models=true)
```bash
python -m src.scripts.training.train --batch_size=1024 --gamma=0.9999995 --optimizer=SGD --lr=1 --n_games=4294967296 --config=4x4_default --name=4x4_default
```

Train pair of networks on 5 x 5 games (might want to set --log_models=true)
```bash
python -m src.scripts.training.train --batch_size=1024 --gamma=0.9999995 --optimizer=SGD --lr=1 --n_games=4294967296 --config=5x5_default --name=5x5_default
```


#### Robustness (independence / consistency)

Comparing a pair of models (model_a must be larger)
```bash
python -m src.scripts.evaluation.evaluate_consistency --model_a=3x3_default --model_b=2x2_default --dataset=2x2_default
```


#### Robustness (subspaces)

Training (a)
```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_hemisphere --name=2x2_hemisphere
```
Training (b)
```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_halfsphere --name=2x2_halfsphere
```
Evaluate (a)
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_hemisphere --dataset=2x2_default
```
Evaluate (b)
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_halfsphere --dataset=2x2_default
```

#### Robustness (loss function)
Training with linear loss
```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_linear_loss --name=2x2_linear_loss
```
Training with ex-post loss
```bash
python -m src.scripts.training.train --batch_size=128 --gamma=0.999999 --optimizer=SGD --lr=0.01 --n_games=1073741824 --config=2x2_expost_loss --name=2x2_expost_loss
```
Evaluate (linear loss)
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_linear_loss --dataset=2x2_default
```
Evaluate (ex-post loss)
```bash
python -m src.scripts.evaluation.evaluate --model=2x2_ex_post_loss --dataset=2x2_default
```


#### Robustness (hyperparaeters)
Online learning 
```bash
python -m src.scripts.training.train --batch_size=1 --gamma=0.999999 --optimizer=SGD --lr=0.01 --n_games=8388608 --config=2x2_default --name=2x2_nobatch
```
Doubled batch size 
```bash
python -m src.scripts.training.train --batch_size=256 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=2147483648 --config=2x2_default --name=2x2_doublebatch
```
No learning rate decay
```bash
python -m src.scripts.training.train --batch_size=1 --gamma=1 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_default --name=2x2_nolrdecay
```


#### Robustness (architecture)
Halved number of neurons
```bash
python -m src.scripts.training.train --batch_size=1 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_halved_neurons --name=2x2_halved_neurons
```
Doubled number of neurons
```bash
python -m src.scripts.training.train --batch_size=1 --gamma=0.999999 --optimizer=SGD --lr=1 --n_games=1073741824 --config=2x2_doubled_neurons --name=2x2_doubled_neurons
```

