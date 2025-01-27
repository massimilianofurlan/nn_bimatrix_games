import argparse
import datetime
import os
import pickle
import json
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from src.utilities.bimatrix_utils import *
from src.utilities.io_utils import save_to_pickle
from src.utilities.data_utils import save_dataset
from src.modules.sampler import BimatrixSampler

def process_batch(batch, n_traces): return [process_game(game, n_traces) for game in batch]

def broadcast(list_arr, func, **kwargs):
        return np.array([func(arr, **kwargs) for arr in list_arr])

 
def process_game(G, n_traces):
    """
    Process a single game to compute various evaluation metrics.
    
    Args:
    - G: tuple of np.ndarray, payoff matrices for two players.

    Returns:

    """
    # mask dominated pure strategies
    dominated_mask1, dominated_min_payoff_diff1 = get_dominated_mask(G[0], extent=True)
    dominated_mask2, dominated_min_payoff_diff2 = get_dominated_mask(G[1].T, extent=True)
    dominated_mask = np.array([dominated_mask1, dominated_mask2])
    dominated_min_payoff_diff = np.array([dominated_min_payoff_diff1, dominated_min_payoff_diff2])
    # mask rationalizable pure strategies (two players finite game -> iesds)
    rationalizable_mask = get_rationalizable_mask(G, dominated_mask=dominated_mask)
    # compute nash equilibria
    set_nash, set_nash_payoffs = get_nash_equilibria(G)
    # mask pure nash equilibria 
    pure_nash_mask = get_pure_nash_mask(set_nash)
    # mask pareto optimal and utilitarian nash equilibria
    pareto_nash_mask, utilitarian_nash_mask, payoff_dominance_mask = get_pareto_optimal_nash_mask(G, set_nash_payoffs)
    # mask harsanyi-selten linear tracing selected nashe quilibria
    harsanyi_selten_mask, harsanyi_selten_traces, harsanyi_selten_traces_reldiff = get_harsanyi_selten_mask(G, set_nash, n_trace=n_traces)
    # compute stability index of each nash equilibrium
    nash_index = get_indeces(G, set_nash)
    # compute minimax strategies and payoffs
    maxmin_payoffs = get_maxmin_payoff(G)

    return (dominated_mask, dominated_min_payoff_diff, rationalizable_mask, set_nash, set_nash_payoffs, 
            pure_nash_mask, pareto_nash_mask, utilitarian_nash_mask, payoff_dominance_mask, harsanyi_selten_mask,
            harsanyi_selten_traces, harsanyi_selten_traces_reldiff, nash_index, maxmin_payoffs)


def label_dataset(dataset, n_traces=1000, n_workers=os.cpu_count(), batch_size=256):
    """
    Label the specified dataset.

    Args:
    - dataset: list of tuples, each containing payoff matrices for two players.
    - n_workers: int, number of workers for parallel processing.
    - batch_size: int, size of each batch for processing.

    Returns:
    - labels: dictionary containing labeled metrics.
    """
    total_games = len(dataset)
    print(f"\nTotal games: {total_games}")

    # Create batches
    batches = [dataset[i:i + batch_size] for i in range(0, total_games, batch_size)]
    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for batch_results in tqdm(executor.map(process_batch, batches, [n_traces]*len(batches)), total=len(batches)):
            all_results.extend(batch_results)

    # Unpack results
    (dominated_mask, dominated_min_payoff_diff, rationalizable_mask, set_nash, set_nash_payoffs, 
     pure_nash_mask, pareto_nash_mask, utilitarian_nash_mask, payoff_dominance_mask, harsanyi_selten_mask,
     harsanyi_selten_traces, harsanyi_selten_traces_reldiff, nash_index, maxmin_payoffs) = zip(*all_results)

    # Create labels dictionary
    labels = {
        'set_nash': set_nash,
        'set_nash_payoffs': set_nash_payoffs,
        'pure_nash_mask': pure_nash_mask,
        'dominated_mask': dominated_mask,
        'dominated_min_payoff_diff': dominated_min_payoff_diff,
        'rationalizable_mask': rationalizable_mask,
        'pareto_nash_mask': pareto_nash_mask,
        'utilitarian_nash_mask': utilitarian_nash_mask,
        'payoff_dominance_mask': payoff_dominance_mask,
        'harsanyi_selten_mask': harsanyi_selten_mask,
        'harsanyi_selten_traces': harsanyi_selten_traces,
        'harsanyi_selten_traces_reldiff': harsanyi_selten_traces_reldiff,
        'nash_index': nash_index,
        'maxmin_payoffs': maxmin_payoffs,
    }

    return labels


def analyze_dataset(labels):
    # Create statistics dictionary with counters
    statistics = {
        'n_nash': broadcast(labels['set_nash'], len),
        'n_pure_nash': broadcast(labels['pure_nash_mask'], np.sum),
        'n_dominated': broadcast(labels['dominated_mask'], np.sum, axis=1),
        'n_rationalizable': broadcast(labels['rationalizable_mask'], np.sum, axis=1),
        'n_pareto_optimal': broadcast(labels['pareto_nash_mask'], np.sum),
        'n_utilitarian': broadcast(labels['utilitarian_nash_mask'], np.sum),            # almost surely unique  
        'n_payoff_dominant': broadcast(labels['payoff_dominance_mask'], np.sum),
        'n_harsanyi_selten': broadcast(labels['harsanyi_selten_mask'], np.sum),         # linear tracing -> almost surely unique
        'n_index_minus' : broadcast(labels['nash_index'], lambda x : np.sum(x == -1)),
        'n_index_zero' : broadcast(labels['nash_index'], lambda x : np.sum(x == 0)),    # almost surely unique  
        'n_index_plus' : broadcast(labels['nash_index'], lambda x : np.sum(x == 1)),
    }
    return statistics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and label evaluation set.')
    parser.add_argument('--n_games', type=int, default=2**17, help='Total number of games to generate and label for the evaluation set. Default is 2^17.')
    parser.add_argument('--n_actions', type=int, default=2, help='Number of actions of each player. Default is 2')
    parser.add_argument('--payoffs_space', type=str, default="sphere_orthogonal", help='Payoffs space')
    parser.add_argument('--game_class', type=str, default="general_sum", help='Class of games. Default is general_sum')
    parser.add_argument('--n_traces', type=int, default=1000, help='Trace lenght for Harsanyi-Selten linear tracing procedure')  
    parser.add_argument('--name', type=str, default=None, help='Dataset name')
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    
    rand_bimatrix = BimatrixSampler(n_actions=args.n_actions, payoffs_space=args.payoffs_space, game_class=args.game_class, dtype=torch.float64)
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"Generating testing set...")
    dataset = rand_bimatrix(args.n_games).numpy()

    print(f"Labeling testing set...")
    labels = label_dataset(dataset, n_traces=args.n_traces)

    print(f"Computing statistics... ")
    statistics = analyze_dataset(labels)
    
    save_dataset(dataset, labels, statistics, timestamp, args)
