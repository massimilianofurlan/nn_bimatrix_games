import time
import argparse
import torch
import numpy as np 
import os 
import sys 
import matplotlib.pyplot as plt
from src.utilities.model_utils import select_models, load_model
from src.utilities.data_utils import select_dataset, print_metadata, load_labels, load_statistics, log_metadata
from src.utilities.eval_utils import transpose_game, get_closest_nash
from src.utilities.io_utils import load_from_pickle, preview_dataset, print_and_log


########################### LOADING STUFF ###########################
#####################################################################

parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
parser.add_argument('--model_a', type=str, default=None, help="Model1 folder")
parser.add_argument('--model_b', type=str, default=None, help="Model2 folder")
parser.add_argument('--dataset', type=str, default=None, help="Dataset Folder (must meet bigger model dim")
args = parser.parse_args()

# setting device and seed
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1)

# load models
os.system('cls' if os.name == 'nt' else 'clear')
model1_a, _, simulation_metadata_a, model_dir = select_models(model_dir=args.model_a, device = device)
model1_b, _, simulation_metadata_b, model_dir = select_models(model_dir=args.model_b, device = device)
model1_a.eval()
model1_b.eval()

# load testing set 
os.system('cls' if os.name == 'nt' else 'clear')
testing_set, dataset_metadata, dataset_dir = select_dataset(dataset_dir=args.dataset)

if simulation_metadata_a['n_actions'] <= simulation_metadata_b['n_actions']:
    print('model a must be bigger than model b')
    exit()

if dataset_metadata['n_actions'] != simulation_metadata_a['n_actions']:
    print('dataset n_actions must be equal to model a n_actions')
    exit()

# evaluation folder
eval_dir = "out"
os.makedirs(eval_dir, exist_ok=True)

# evaluation file
eval_file = f'{eval_dir}/{args.model_a}_{args.model_b}_consistency.txt'

# loading statistics and evaluation 
os.system('cls' if os.name == 'nt' else 'clear')
print(f"Loading labels... ")
labels = load_labels(dataset_dir)
print(f"Loading statistics... ")
statistics = load_statistics(dataset_dir)

# convert games to tensor
#games = np.array(testing_set[gamma_nash_mask], dtype=np.float32)
games = np.array(testing_set, dtype=np.float32)
games = torch.tensor(games, device=device, dtype=torch.float32, requires_grad=False)
n_games, n_players, n_actions, _ = games.size()

# visualize to terminal
print(f"\nModel a: ")
print_metadata(simulation_metadata_a)
print(f"\nModel b: ")
print_metadata(simulation_metadata_b)
print(f"\nEvaluating on: ")
preview_dataset(dataset_metadata, testing_set)
print()

# log to file 
log_metadata(eval_file, simulation_metadata_a, "Model a: ")
log_metadata(eval_file, simulation_metadata_b, "Model b: ")
log_metadata(eval_file, dataset_metadata, "Dataset: ", 'a')


############################ DEFINITIONS ############################
#####################################################################

quantiles = torch.tensor([0.25, 0.5, 0.75, 0.90, 0.95, 0.99], device=device)

n_actions_a = simulation_metadata_a['n_actions']
n_actions_b = simulation_metadata_b['n_actions']
delta_n_actions = n_actions_a - n_actions_b
dominated_mask = torch.tensor(np.array(labels['dominated_mask']), device=device)
dominance_reduced_mask = np.all(statistics['n_dominated'] == [delta_n_actions,delta_n_actions], axis=1)
n_games = dominance_reduced_mask.sum()

undominated_mask = ~dominated_mask[dominance_reduced_mask]
undominated_a_mask = (undominated_mask[:,0,:]).unsqueeze(2) & (undominated_mask[:,1,:]).unsqueeze(1)
undominated_a_mask = undominated_a_mask.unsqueeze(1).expand(n_games,2,n_actions_a,n_actions_a)
games_a = games[dominance_reduced_mask]
games_b = games_a.masked_select(undominated_a_mask).view(n_games, 2, n_actions_b, n_actions_b)


batch_size = 2**11
with torch.no_grad():
    strategies_b = torch.empty(n_games, n_actions_b, device=device)
    strategies_a = torch.empty(n_games, n_actions_a, device=device)
    start_time = time.time()
    for start_index in range(0, n_games, batch_size):
        end_index = start_index + min(batch_size, n_games - start_index)
        G_b = games_b[start_index:end_index]
        G_a = games_a[start_index:end_index]
        #
        p_b = model1_b(G_b)
        p_a = model1_a(G_a)
        #
        strategies_b[start_index:end_index] = p_b
        strategies_a[start_index:end_index] = p_a
        #
        progress_percentage = (end_index / n_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    strategies_a_original = strategies_a.clone()
    strategies_a = strategies_a.masked_select(undominated_mask[:,0,:]).view(n_games, n_actions_b)
    distance = torch.sum(torch.abs(strategies_b - strategies_a),dim=1)/2


def quantiles_string(arr, quants = quantiles.cpu().numpy()):
    return np.array2string(np.quantile(arr.cpu().numpy(), quants), formatter={'float_kind': lambda x: f"{x:.3f}"})

f = open(eval_file, 'a')

def print_evaluation_results(distance, mask):
    avg_distance = distance[mask].mean()
    std_distance = distance[mask].std()
    quantiles_distance = torch.quantile(distance[mask], quantiles)
    high_distance_mask = avg_distance > quantiles_distance[-1]
    print_and_log(f'Number of Games: {np.sum(mask)}', f)
    print_and_log(f'Average Distance: {avg_distance:.3f} ({std_distance:.3f})', f)
    print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(distance[mask])}', f)
    print_and_log('', f)


zero_pure_nash_mask = statistics['n_pure_nash'][dominance_reduced_mask] == 0
one_pure_nash_mask = statistics['n_pure_nash'][dominance_reduced_mask] == 1
two_pure_nash_mask = statistics['n_pure_nash'][dominance_reduced_mask] == 2

print()
print_and_log('ALL GAMES', f)
print_evaluation_results(distance, np.ones(n_games,dtype=bool))
print_and_log('0 PURE NASH EQUILIBRIA', f)
print_evaluation_results(distance, zero_pure_nash_mask)
print_and_log('1 PURE NASH EQUILIBRIA', f)
print_evaluation_results(distance, one_pure_nash_mask)
print_and_log('2 PURE NASH EQUILIBRIA', f)
print_evaluation_results(distance, two_pure_nash_mask)


'''
def get_closest_nash(strategies: torch.Tensor, set_nash):
    # find the closest Nash equilibrium and compute the distances for each game in the batch.
    device = strategies.device
    max_n_nash = torch.max(torch.tensor([arr.shape[0] for arr in set_nash], device=device))
    # expand strategy_profile for broadcasting
    strategy_profile_expanded = strategies.unsqueeze(1).expand(-1, max_n_nash, -1, -1)
    # stack set_nash tensors and pad them
    set_nash_tensor = [torch.tensor(np.float32(arr), dtype=torch.float32, device=device) for arr in set_nash]
    set_nash_expanded = torch.nn.utils.rnn.pad_sequence(set_nash_tensor, batch_first=True, padding_value=float('inf'))
    # compute distances (maximum supremum norm across agents)
    #dists = torch.sum(torch.abs(set_nash_expanded - strategy_profile_expanded), dim=(2, 3)) / 2
    dists = torch.amax(torch.abs(set_nash_expanded - strategy_profile_expanded), dim=(2, 3))
    min_dist, argmin_dist = dists.min(dim=1)
    return argmin_dist.detach().cpu().numpy(), min_dist.detach().cpu().numpy()

set_nash = [labels['set_nash'][z] for z in range(0,len(labels['set_nash'])) if submask[z]]
idx_nash_a, dist_nash_a = get_closest_nash(strategies_a_original,set_nash)
strategies_b_expanded = torch.zeros(n_games, n_players, 3, device=device)
strategies_b_expanded[undominated_mask] = strategies_b.view(-1)[:torch.sum(undominated_mask)]
idx_nash_b, dist_nash_b = get_closest_nash(strategies_b_expanded,set_nash)

# dist closest nash
dist_nash_a.mean()
dist_nash_b.mean()
# freq dist closest nash < 0.1
(dist_nash_a < 0.1).mean()
(dist_nash_b < 0.1).mean()
# same nash
np.mean(idx_nash_a == idx_nash_b)
'''

