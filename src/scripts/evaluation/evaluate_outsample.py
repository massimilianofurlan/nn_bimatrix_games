import argparse
import sys
import os 
import torch
import numpy as np
from src.scripts.evaluation.evaluate import *

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games perturbed by random affine transformations")
    parser.add_argument('--model', type=str, default=None, help="Model folder")
    parser.add_argument('--dataset', type=str, default=None, help="Dataset Folder")
    # Process configs
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.manual_seed(1)

    # load models
    os.system('cls' if os.name == 'nt' else 'clear')
    model1, model2, simulation_metadata, model_dir = select_models(model_dir = args.model, device=device)
    model1.eval()
    model2.eval()

    # load testing set 
    os.system('cls' if os.name == 'nt' else 'clear')
    testing_set, dataset_metadata, dataset_dir = select_dataset(dataset_dir=args.dataset)
    
    n_games, n_players, n_actions, _ = testing_set.shape
    a = np.random.rand(n_games, n_players) * n_actions * 2 - n_actions
    b = np.random.rand(n_games, n_players) * (n_actions-1.0) + 1.0
    a = a.reshape(n_games, n_players, 1, 1)
    b = b.reshape(n_games, n_players, 1, 1)
    testing_set = a + b * testing_set 

    # load labels
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Loading labels... ")
    labels = load_labels(dataset_dir)
    # load statistics
    print(f"Loading statistics... ")
    statistics = load_statistics(dataset_dir)
    print(f"All done")

    # visualize to terminal
    print(f"\nModel: ")
    print_metadata(**simulation_metadata)
    print(f"\nEvaluating on: ")
    preview_dataset(dataset_metadata, testing_set)

    # generate evaluation folder
    eval_dir = os.path.join("models", model_dir, dataset_dir)
    os.makedirs(eval_dir, exist_ok=True)
    eval_file = f'{eval_dir}/evaluation_outsample.txt'
    log_metadata(eval_file, simulation_metadata, "Models: ")
    log_metadata(eval_file, dataset_metadata, "Dataset: ", 'a')

    f = open(eval_file, 'a')

    print('\nTesting model...')
    evaluation_output = evaluate(model1, model2, testing_set, labels, device)
    
    # compute distance from nxn_default
    n_actions = dataset_metadata['n_actions']
    eval_dir_nxn_default = os.path.join('models',f'{n_actions}x{n_actions}_default',dataset_dir,'evaluation_output.pkl')
    dist_from_default = np.ones(len(testing_set))*np.inf
    if os.path.exists(eval_dir_nxn_default):
        evaluation_nxn_default = load_from_pickle(eval_dir_nxn_default)
        strategy_profiles_nxn_default = evaluation_nxn_default['strategy_profiles']
        strategy_profiles = evaluation_output['strategy_profiles']
        dist_from_default = np.amax(np.sum(np.abs(strategy_profiles_nxn_default - strategy_profiles), axis=2), axis=1) * 0.5
    evaluation_output['dist_from_default'] = dist_from_default
    
    print_and_log('\n\n[[[ ALL GAMES ]]]', f)
    all_games = np.ones(len(testing_set), dtype=bool)
    print_evaluation_results(evaluation_output, statistics, all_games, f=f)

    print_and_log('\n\n[[[ DOMINANCE SOLVABLE]]]', f)
    dominance_solvable_mask = np.all(statistics['n_rationalizable'] == 1, axis=1)
    print_evaluation_results(evaluation_output, statistics, dominance_solvable_mask, f=f)

    print_and_log('\n\n[[[ 0 PURE NASH EQUILIBRIA ]]]', f)
    zero_pure_nash_mask = statistics['n_pure_nash'] == 0
    print_evaluation_results(evaluation_output, statistics, zero_pure_nash_mask, f=f)

    print_and_log('\n\n[[[ 1 PURE NASH EQUILIBRIUM ]]]', f)
    one_pure_nash_mask = statistics['n_pure_nash'] == 1
    print_evaluation_results(evaluation_output, statistics, one_pure_nash_mask, f=f)

    print_and_log('\n\n[[[ > 1 PURE NASH EQUILIBRIA ]]]', f)
    multiple_pure_nash_mask = statistics['n_pure_nash'] > 1
    print_evaluation_results(evaluation_output, statistics, multiple_pure_nash_mask, f=f)

    print_and_log('\n\n[[[ >= 1 PURE NASH EQUILIBRIA ]]]', f)
    some_pure_nash_mask = np.logical_or(one_pure_nash_mask, multiple_pure_nash_mask)
    print_evaluation_results(evaluation_output, statistics, some_pure_nash_mask, f=f)

    f.close()


if __name__ == "__main__":
    main()