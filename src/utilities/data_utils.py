import os
import json
import pickle
import torch
import re
import numpy as np

from src.utilities.io_utils import *

## Utilities to load datasets

def select_dataset(dataset_dir=None):
    """Allow user to select a dataset based on dataset_dir or load its training set."""
    if dataset_dir:
        dataset_dir = dataset_dir
        dataset_metadata = load_metadata("data", dataset_dir, 'metadata.json')
    else:
        dataset_info = display_info('data', 'metadata.json')
        dataset_dir, dataset_metadata = select_item(dataset_info)
    dataset = load_dataset(dataset_dir)
    return dataset, dataset_metadata, dataset_dir

def load_dataset(dataset_dir, base_dir="data"):
    """Load a specific set (e.g., training) from a dataset directory."""
    dataset_path = os.path.join(base_dir, dataset_dir, 'dataset.pkl')
    dataset = load_from_pickle(dataset_path)
    return dataset

def load_labels(dataset_dir, base_dir="data"):
    """Load a labels from a dataset directory."""
    labels_path = os.path.join(base_dir, dataset_dir, 'labels.pkl')
    labels = load_from_pickle(labels_path)
    return labels

def load_statistics(dataset_dir, base_dir="data"):
    """Load a statistics from a dataset directory."""
    statistics_path = os.path.join(base_dir, dataset_dir, 'statistics.pkl')
    statistics = load_from_pickle(statistics_path)
    return statistics

def load_evaluation(model_dir, dataset_dir, base_dir="models"):
    """Load a evaluation from a model/dataset directory."""
    evaluation_path = os.path.join(base_dir, model_dir, dataset_dir, 'evaluation_output.pkl')
    evaluation = load_from_pickle(evaluation_path)
    return evaluation

def save_dataset(dataset, labels, statistics, timestamp, args, save_path = None):
    """
    Save the generated dataset and labels using pickle.

    Args:
    - dataset: list of tuples, each containing payoff matrices for two players.
    - labels: dictionary containing labeled metrics.
    - statistics: dictionary containing dataset statistics.
    - timestamp: str, timestamp identifying the dataset.
    - args: str, input arguments.
    """
    datest_name = args.name if args.name else timestamp
    dataset_folder = save_path if save_path else os.path.join("data", datest_name)
    os.makedirs(dataset_folder, exist_ok=True)

    # Save dataset
    dataset_filename = os.path.join(dataset_folder, "dataset.pkl")
    save_to_pickle(dataset, dataset_filename)
    print(f"Dataset saved to '{dataset_filename}'")

    # Save labels
    labels_filename = os.path.join(dataset_folder, "labels.pkl")
    save_to_pickle(labels, labels_filename)
    print(f"Labels saved to '{labels_filename}'")

    # Save statistics
    statistics_filename = os.path.join(dataset_folder, "statistics.pkl")
    save_to_pickle(statistics, statistics_filename)
    print(f"Statistics saved to '{statistics_filename}'")

    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'n_games': args.n_games,
        'n_actions': args.n_actions,
        'payoffs_space': args.payoffs_space,
        'game_class': args.game_class,
        'n_traces': args.n_traces
    }
    metadata_filename = os.path.join(dataset_folder, "metadata.json")
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to '{metadata_filename}'")

def read_games_from_file(file_path):
    """
    Reads games from file.
    """
    games_data = {}
    current_game = ""
    with open(file_path, 'r') as file:
        for line in file.read().strip().split("\n"):
            if line.startswith("#"):
                current_game = line[2:].strip()
                games_data[current_game] = []
            else:
                # Directly parse the line within this loop
                elements = line.replace('(', '').replace(')', '').split()
                outcomes = [tuple(float(num) for num in element.split(',')) for element in elements]
                games_data[current_game].append(outcomes)
     
    games_matrices = {}
    for game, outcomes in games_data.items():
        player1_matrix = np.array([[outcome[0] for outcome in row if outcome] for row in outcomes if any(row)])
        player2_matrix = np.array([[outcome[1] for outcome in row if outcome] for row in outcomes if any(row)])
        games_matrices[game] = (player1_matrix, player2_matrix)
    
    return games_matrices
