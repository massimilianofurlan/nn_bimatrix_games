## Overview

This repository contains the code for [*"D. Condorelli, M. Furlan. Deep Learning to Play Games"*](https://arxiv.org/pdf/2409.15197). 

Replication instructions are listed in the [replication](replication.md) file. _This is a work in progress. Replication instructions might be broken._ 

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation (one-time setup)

Follow these steps to set up the project environment. You only need to do this once:

1. **Clone the Repository:**  
   Download the code to your local machine:

    ```bash
    git clone https://github.com/massimilianofurlan/nn_bimatrix_games.git
    ```

2. **Install Python:**  
   Make sure Python is installed. On macOS, you can use Homebrew:

    ```bash
    brew install python3.12
    ```

3. **Set Up a Virtual Environment:**  
   Navigate to the project folder and create a virtual environment:

    ```bash
    cd nn_bimatrix_games
    python3.12 -m venv venv
    ```

4. **Activate the Virtual Environment:**  
   Activate the environment to isolate project dependencies:

    ```bash
    source venv/bin/activate
    ```

5. **Install Dependencies:**  
   Upgrade `pip` and install the required libraries:

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

Once these steps are completed, the environment is ready for use.

## Basic Usage

Make sure you are in the `nn_bimatrix_games` project folder and the virtual environment is activated:

```bash
cd nn_bimatrix_games
source venv/bin/activate
```

### (a) Launching Training

To train a pair of neural networks on 2×2 games, use the following command:

```bash
python -m src.scripts.training.train --config=2x2_example --batch_size=128 --n_games=33554432 --name=2x2_example
```

---

### (b) Generating an Evaluation Set

To generate a labeled dataset for evaluating 2×2 games:

```bash
python -m src.scripts.data.generate_evalset --n_actions=2 --n_games=131072 --name=2x2_example_dataset
```

---

### (c) Evaluating the Model on the Evaluation Set

To evaluate a trained model on the evaluation dataset:

```bash
python -m src.scripts.evaluation.evaluate --model=2x2_example --dataset=2x2_example_dataset
```

> **Note:** Make sure that the `--model` and `--dataset` names match the names used during training and evaluation set generation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.