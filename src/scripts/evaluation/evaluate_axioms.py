import argparse
import sys
import torch
import itertools
import math
import numpy as np
from src.utilities.model_utils import *
from src.utilities.data_utils import *
from src.utilities.eval_utils import *
from src.utilities.viz_utils import *
from src.utilities.io_utils import *

parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
parser.add_argument('--model', type=str, default=None, help="Model folder")
parser.add_argument('--dataset', type=str, default=None, help="Dataset Folder")
# Process configs
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1)

# load simulation_metadata
os.system('cls' if os.name == 'nt' else 'clear')
model1, model2, simulation_metadata, model_dir = select_models(model_dir = args.model, device=device)
model1.eval()
model2.eval()

# load testing set 
os.system('cls' if os.name == 'nt' else 'clear')
testing_set, dataset_metadata, dataset_dir = select_dataset(dataset_dir=args.dataset)

# visualize to terminal
print(f"\nModel: ")
print_metadata(simulation_metadata)
print(f"\nEvaluating on: ")
preview_dataset(dataset_metadata, testing_set)

# evaluation folder
eval_dir = os.path.join("models", model_dir, dataset_dir)
if not os.path.exists(eval_dir):
    print(f"Error: Directory '{eval_dir}' does not exist.")
    sys.exit(1)

# evaluation file
eval_file = f'{eval_dir}/evaluation_axioms.txt'
log_metadata(eval_file, simulation_metadata, "Model: ")
log_metadata(eval_file, dataset_metadata, "Dataset: ", 'a')

# load labels
print(f"\nLoading labels... ")
labels = load_labels(dataset_dir)
# load statistics
print(f"Loading statistics... ")
statistics = load_statistics(dataset_dir)
# load evaluation
print(f"Loading evaluation... ")
evaluation_output = load_from_pickle(f'{eval_dir}/evaluation_output.pkl')
# all done
print(f"All done")


# convert games to tensor
#games = np.array(testing_set[gamma_nash_mask], dtype=np.float32)
games = np.array(testing_set, dtype=np.float32)
games = torch.tensor(games, device=device, dtype=torch.float32, requires_grad=False)
n_games, n_players, n_actions, _ = games.size()


############################# DEFINITIONS ##############################
#####################################################################

batch_size = 2**14
quantiles = torch.tensor([0.25, 0.5, 0.75, 0.90, 0.95, 0.99], device=device)

harsanyi_selten_mask = labels['harsanyi_selten_mask']
harsanyi_selten_traces = labels['harsanyi_selten_traces']
unique_nash_mask = statistics['n_nash'] == 0
multiple_nash_mask = statistics['n_nash'] > 1

print('\nTesting model...')

############################ PERMUTATIONS ###########################
#####################################################################

print('\nTest 1/4 - Invariance to Renaming of Choiches ...')

def permute_games(game_batch, perms):
    p_idx = 0
    game_batch_permutations = []
    for perm1 in perms:
        for perm2 in perms:
            permuted_game = game_batch[:, :, :, perm1][:, :, perm2]
            game_batch_permutations.append(permuted_game)
            p_idx += 1
    game_batch_expanded = torch.stack(game_batch_permutations, dim=0)
    return game_batch_expanded

def invert_strategy(strategy, inv_perms):
    p_idx = 0
    for perm1 in inv_perms:
        for perm2 in inv_perms:
            strategy[p_idx,:,:] = strategy[p_idx, :, perm2]
            p_idx += 1
    return strategy

n_perms = math.factorial(n_actions)**n_players
n_extended_games = n_games * n_perms
with torch.no_grad():
    perms = torch.tensor(list(itertools.permutations(range(n_actions))),device=device)
    inv_perms = torch.argsort(perms, dim=1)
    games_extended = permute_games(games, perms)
    games_extended = games_extended.view(n_extended_games, n_players, n_actions, n_actions)
    strategies = torch.empty(n_extended_games, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_extended_games, batch_size):
        end_index = start_index + min(batch_size, n_extended_games - start_index)
        G = games_extended[start_index:end_index]
        #
        p = model1(G)
        strategies[start_index:end_index] = p
        #
        progress_percentage = (end_index / n_extended_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    # strategies has shape n_perms x n_games x n_actions
    strategies = strategies.view(n_perms, n_games, n_actions)
    # invert permutation (by reference)
    strategies = invert_strategy(strategies, inv_perms)
    # centroid strategies
    strategy_centroid = strategies.mean(dim=0, keepdim=True)
    
    permutation_distance = torch.sum(torch.abs(strategies - strategy_centroid), dim=2)/2
    permutation_avg_distance = permutation_distance.mean(dim=0)
    permutation_avg_avg_distance = permutation_avg_distance.mean()
    permutation_std_avg_distance = permutation_avg_distance.std()
    permutation_quantiles_avg_distance = torch.quantile(permutation_avg_distance, quantiles)
    permutation_high_avg_distance_mask = permutation_avg_distance > permutation_quantiles_avg_distance[-1]


############################# SYMMETRY ##############################
#####################################################################

print('\nTest 2/4 - Invariance to Renaming of Players ...')

with torch.no_grad():
    p = torch.empty(n_games, n_players, n_actions, device=device)
    #q = torch.empty(n_games, n_players, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_games, batch_size):
        end_index = start_index + min(batch_size, n_games - start_index)
        G = games[start_index:end_index]
        # 
        p1 = model1(G)
        p2 = model2(G)
        #
        p[start_index:end_index] = torch.stack([p1,p2],dim=1)
        #q[start_index:end_index] = torch.stack([q1,q2],dim=1)
        #
        progress_percentage = (end_index / n_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    symmetry_distance = torch.sum(torch.abs(p.diff(dim=1).squeeze(1)),dim=1) / 2
    symmetry_avg_distance = symmetry_distance.mean()
    symmetry_std_distance = symmetry_distance.std()
    symmetry_quantiles_distance = torch.quantile(symmetry_distance,quantiles)
    symmetry_high_distance_mask = symmetry_distance > symmetry_quantiles_distance[-1]


##################### BEST REPLY INVARIANCE (2) #####################
#####################################################################

print('\nTest 3/4 - Invariance to Affine Best Reply Structure Preserving Transformations ...')

def rand_affine_bestreply_preserving_transformation(game_batch, n_transf, device = 'cpu'):
    # generate random best reply preserving trasformation  a_j + b u_i( . ,j)
    batch_size, n_players, n_actions, _ = game_batch.shape
    a1 = torch.rand(n_transf, batch_size, n_actions, device = device) * n_actions * 2
    a1 = a1.unsqueeze(-1).expand(n_transf, batch_size, n_actions, n_actions)
    a1 = a1.permute(0,1,3,2)
    a2 = torch.rand(n_transf, batch_size, n_actions, device = device) * n_actions * 2
    a2 = a2.unsqueeze(-1).expand(n_transf, batch_size, n_actions, n_actions)
    a = torch.stack([a1,a2],dim=2)
    b = torch.rand(n_transf, batch_size, n_players, device = device) * (n_actions-1.0) + 1.0
    b = b.view(n_transf, batch_size, n_players, 1, 1)
    return a + b * game_batch 

n_transf = 64
n_extended_games = n_games * n_transf
with torch.no_grad():
    games_extended = rand_affine_bestreply_preserving_transformation(games, n_transf, device=device)
    games_extended = games_extended.view(n_extended_games, n_players, n_actions, n_actions)
    strategies = torch.empty(n_extended_games, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_extended_games, batch_size):
        end_index = start_index + min(batch_size, n_extended_games - start_index)
        G = games_extended[start_index:end_index]
        #
        p = model1(G)
        strategies[start_index:end_index] = p
        #
        progress_percentage = (end_index / n_extended_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    # strategies has shape n_transf x n_games x n_actions
    strategies = strategies.view(n_transf, n_games, n_actions)
    # centroid strategies
    strategy_centroid = strategies.mean(dim=0, keepdims=True)
    
    affine_bestreply_distance  = torch.sum(torch.abs(strategies - strategy_centroid), dim=2)/2
    affine_bestreply_avg_distance = affine_bestreply_distance.mean(dim=0)
    affine_bestreply_avg_avg_distance = affine_bestreply_avg_distance.mean()
    affine_bestreply_std_avg_distance = affine_bestreply_avg_distance.std()
    affine_bestreply_quantiles_avg_distance = torch.quantile(affine_bestreply_avg_distance, quantiles)
    affine_bestreply_high_avg_distance_mask = affine_bestreply_avg_distance > affine_bestreply_quantiles_avg_distance[-1]


###########################  MONOTONICITY ###########################
#####################################################################

print('\nTest 4/4 - Monotonicity ...')

def rand_sum_to_eq(game_batch, nash_mask, n_transf, device = 'cpu'):
    # generate random best reply preserving trasformation  a_j + b u_i( . ,j)
    batch_size, n_players, n_actions, _ = game_batch.shape
    k = torch.rand(n_transf, batch_size, n_players, device = device) * n_actions * 2
    k = k.unsqueeze(3).unsqueeze(4).expand(n_transf, batch_size, n_players, n_actions, n_actions)
    z = k * nash_mask.unsqueeze(0).unsqueeze(2)
    return game_batch + z

closest_nash_idx = evaluation_output['closest_nash_idx']
closest_nash_distance = evaluation_output['closest_nash_distance']
set_nash = labels['set_nash']
pure_nash_mask = labels['pure_nash_mask']

closest_nash = np.array([set_nash[i][closest_nash_idx[i]] for i in range(n_games)], dtype=np.float32)
closest_nash_mask = np.einsum('ij,ik->ijk', closest_nash[:,0,:], closest_nash[:,1,:])
closest_nash_mask = torch.tensor(closest_nash_mask, device=device)

closest_nash_is_pure = np.array([pure_nash_mask[i][closest_nash_idx[i]] for i in range(n_games)])
is_gamma_nash = closest_nash_distance < 0.05
is_pure_and_gamma_nash = is_gamma_nash & closest_nash_is_pure

games_ = games[is_pure_and_gamma_nash]
closest_nash_mask = closest_nash_mask[is_pure_and_gamma_nash]
n_games_ = games_.shape[0]
n_transf = 64
n_extended_games = n_games_ * n_transf

with torch.no_grad():
    games_extended = rand_sum_to_eq(games_, closest_nash_mask, n_transf, device=device)
    games_extended = games_extended.view(n_extended_games, n_players, n_actions, n_actions)
    strategies = torch.empty(n_extended_games, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_extended_games, batch_size):
        end_index = start_index + min(batch_size, n_extended_games - start_index)
        G = games_extended[start_index:end_index]
        #
        p = model1(G)
        strategies[start_index:end_index] = p
        #
        progress_percentage = (end_index / n_extended_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    # strategies has shape n_transf x n_games x n_actions
    strategies = strategies.view(n_transf, n_games_, n_actions)
    # centroid strategies
    strategy_centroid = strategies.mean(dim=0, keepdims=True)

    monotonicity_distance  = torch.sum(torch.abs(strategies - strategy_centroid), dim=2)/2
    monotonicity_avg_distance = monotonicity_distance.mean(dim=0)
    monotonicity_avg_avg_distance = monotonicity_avg_distance.mean()
    monotonicity_std_avg_distance = monotonicity_avg_distance.std()
    monotonicity_quantiles_avg_distance = torch.quantile(monotonicity_avg_distance, quantiles)
    monotonicity_high_avg_distance_mask = monotonicity_avg_distance > monotonicity_quantiles_avg_distance[-1]


###################### OUTPUT TO FILE AND TERM ######################
#####################################################################

def quantiles_string(arr, quants = quantiles.cpu().numpy()):
    return np.array2string(np.quantile(arr.cpu().numpy(), quants), formatter={'float_kind': lambda x: f"{x:.3f}"})

f = open(eval_file, 'a')

print()
print_and_log('PERMUTATIONS',f)
print_and_log(f'Average Distance: {permutation_avg_avg_distance:.3f} ({permutation_std_avg_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(permutation_quantiles_avg_distance)}', f)
print_and_log('', f)

print_and_log('SYMMETRY',f)
print_and_log(f'Average Distance: {symmetry_avg_distance:.3f} ({symmetry_std_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(symmetry_quantiles_distance)}', f)
print_and_log('', f)


print_and_log('AFFINE BEST REPLY PRESERVING TRANSFORMATIONS',f)
print_and_log(f'Average Distance: {affine_bestreply_avg_avg_distance:.3f} ({affine_bestreply_std_avg_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(affine_bestreply_quantiles_avg_distance)}', f)
print_and_log('', f)

print_and_log('MONOTONICITY',f)
print_and_log(f'Average Distance (0.05-Pure Nash): {monotonicity_avg_avg_distance:.3f} ({monotonicity_std_avg_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(monotonicity_quantiles_avg_distance)}', f)


'''
############################# UNANIMITY #############################
#####################################################################

print('\nTest 4/4 - Unanimity  ...')

argmax_profiles = games == torch.amax(games, dim=(2,3), keepdim=True)
joint_argmax_profile = argmax_profiles.prod(dim=1)
joint_argmax_profile_exists_mask = joint_argmax_profile.any(dim=(1,2)).cpu().numpy()
multiple_nash_and_joint_argmax_exists_mask = np.logical_and(joint_argmax_profile_exists_mask, multiple_nash_mask)

argmax_strategy1 = argmax_profiles[:,0,:,:].sum(dim=2)
argmax_strategy2 = argmax_profiles[:,1,:,:].sum(dim=1)
argmax_strategy_profiles = torch.stack([argmax_strategy1,argmax_strategy2],dim=1)

set_nash = labels['set_nash']
harsanyi_selten_nash = np.array([set_nash[i][mask] for i,mask in enumerate(harsanyi_selten_mask)], dtype=np.float32)
harsanyi_selten_nash = torch.tensor(harsanyi_selten_nash, device= device).squeeze(1)
harsanyi_selten_and_unanimous_mask = torch.all(harsanyi_selten_nash[multiple_nash_and_joint_argmax_exists_mask] == argmax_strategy_profiles[multiple_nash_and_joint_argmax_exists_mask], dim=(1,2))
harsanyi_selten_and_notunanimous_mask = ~harsanyi_selten_and_unanimous_mask.cpu().numpy()
harsanyi_selten_and_notunanimous_freq = harsanyi_selten_and_notunanimous_mask.mean()

with torch.no_grad():
    strategy_profiles = torch.empty(n_games, n_players, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_games, batch_size):
        end_index = start_index + min(batch_size, n_games - start_index)
        G = games[start_index:end_index]
        G_transpose = transpose_game(G)
        #
        p = model1(G)
        q = model2(G_transpose)
        strategy_profiles[start_index:end_index] = torch.stack([p,q],dim=1)
        #
        progress_percentage = (end_index / n_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    unanimity_distance  = torch.amax(torch.sum(torch.abs(strategy_profiles - argmax_strategy_profiles), dim=2)/2,dim=1)
    unanimity_distance = unanimity_distance[multiple_nash_and_joint_argmax_exists_mask]
    unanimity_avg_distance = unanimity_distance.mean()
    unanimity_std_distance = unanimity_distance.std()
    unanimity_quantiles_distance = torch.quantile(unanimity_distance, quantiles)
    unanimity_high_distance_mask = unanimity_distance > unanimity_quantiles_distance[-1]



################## POSITIVE AFFINE TRANSFORMATIONS ##################
#####################################################################
print('\nTest 3/7 - Invariance to Positive Affine Transformations ...')

def rand_affine_transformation(game_batch, n_transf, device = 'cpu'):
    # generate random positive affine transformation a \in [-n_actions, n_actions], b \in [1,n_actions]
    batch_size, n_players, n_actions, _ = game_batch.shape
    a = torch.rand(n_transf, batch_size, n_players, device = device) * n_actions * 2 - n_actions
    b = torch.rand(n_transf, batch_size, n_players, device = device) * (n_actions-1.0) + 1.0
    a = a.view(n_transf, batch_size, n_players, 1, 1)
    b = b.view(n_transf, batch_size, n_players, 1, 1)
    return a + b * game_batch 

n_transf = 64
n_extended_games = n_games * n_transf
with torch.no_grad():
    games_extended = rand_affine_transformation(games, n_transf, device=device)
    games_extended = games_extended.view(n_extended_games, n_players, n_actions, n_actions)
    strategy_profiles = torch.empty(n_extended_games, n_players, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_extended_games, batch_size):
        end_index = start_index + min(batch_size, n_extended_games - start_index)
        G = games_extended[start_index:end_index]
        G_transpose = transpose_game(G)
        #
        p = model1(G)
        q = model2(G_transpose)
        strategy_profiles[start_index:end_index] = torch.stack([p,q],dim=1)
        #
        progress_percentage = (end_index / n_extended_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    # strategy profiles has shape n_transf x n_games x n_players x n_actions
    strategy_profiles = strategy_profiles.view(n_transf, n_games, n_players, n_actions)
    
    strategy_profile_centroid = strategy_profiles.mean(dim=0, keepdims=True)    

    affine_distance  = torch.amax(torch.sum(torch.abs(strategy_profiles - strategy_profile_centroid), dim=3)/2,dim=2)
    affine_avg_distance = affine_distance.mean(dim=0)
    affine_avg_avg_distance = affine_avg_distance.mean()
    affine_std_avg_distance = affine_avg_distance.std()
    affine_quantiles_avg_distance = torch.quantile(affine_avg_distance,quantiles)
    affine_high_avg_distance_mask = affine_avg_distance > affine_quantiles_avg_distance[-1]


####################### BEST REPLY INVARIANCE #######################
#####################################################################

print('\nTest 4/7 - Invariance to Best Reply Structure Preserving Transformations  ...')

def rand_bestreply_preserving_transformation(game_batch, n_transf, device = 'cpu'):
    # generate random best reply preserving trasformation u_i( . ,x_j) + delta_j
    batch_size, n_players, n_actions, _ = game_batch.shape
    a = torch.rand(n_transf, batch_size, n_actions, device = device) * n_actions * 2
    a = a.unsqueeze(-1).expand(n_transf, batch_size, n_actions, n_actions)
    a = a.permute(0,1,3,2)
    b = torch.rand(n_transf, batch_size, n_actions, device = device) * n_actions * 2
    b = b.unsqueeze(-1).expand(n_transf, batch_size, n_actions, n_actions)
    a = torch.stack([a,b],dim=2)
    return a + game_batch

n_transf = 64
n_extended_games = n_games * n_transf
with torch.no_grad():
    games_extended = rand_bestreply_preserving_transformation(games, n_transf, device=device)
    games_extended = games_extended.view(n_extended_games, n_players, n_actions, n_actions)
    strategy_profiles = torch.empty(n_extended_games, n_players, n_actions, device=device)
    start_time = time.time()
    for start_index in range(0, n_extended_games, batch_size):
        end_index = start_index + min(batch_size, n_extended_games - start_index)
        G = games_extended[start_index:end_index]
        G_transpose = transpose_game(G)
        #
        p = model1(G)
        q = model2(G_transpose)
        strategy_profiles[start_index:end_index] = torch.stack([p,q],dim=1)
        #
        progress_percentage = (end_index / n_extended_games) * 100 
        print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)
    
    # strategy profiles has shape n_transf x n_games x n_players x n_actions
    strategy_profiles = strategy_profiles.view(n_transf, n_games, n_players, n_actions)
    
    strategy_profile_centroid = strategy_profiles.mean(dim=0, keepdims=True)
    
    bestreply_distance = torch.amax(torch.sum(torch.abs(strategy_profiles - strategy_profile_centroid), dim=3)/2,dim=2)
    bestreply_avg_distance = bestreply_distance.mean(dim=0)
    bestreply_avg_avg_distance = bestreply_avg_distance.mean()
    bestreply_std_avg_distance = bestreply_avg_distance.std()
    bestreply_quantiles_avg_distance = torch.quantile(bestreply_avg_distance, quantiles)
    bestreply_high_avg_distance_mask = bestreply_avg_distance > bestreply_quantiles_avg_distance[-1]


######################## IRRELEVANT ACTIONS #########################
#####################################################################

print('\nTest 5/5 - Independence Strategically Irrelevant Actions  ...')

def border_game(game_batch, n_actions, n_actions_model):
    # border n_actions x n_actions game with a strictly dominated strategy 
    # to match n_actions_model x n_actions_model 
    # action in position 0 will dominate new actions. 
    if n_actions == n_actions_model:
        return game_batch
    else:
        device = game_batch.device
        batch_size = game_batch.shape[0]
        game_batch_ = torch.zeros(batch_size, 2, n_actions_model, n_actions_model, dtype=torch.float32, device = device)
        game_batch_[:, :, 0:n_actions, 0:n_actions] = game_batch
        game_batch_[:, 0, n_actions:, :] = game_batch_[:, 0, 0, :].unsqueeze(1) - n_actions/4 
        game_batch_[:, 1, :, n_actions:] = game_batch_[:, 1, :, 0].unsqueeze(2) - n_actions/4
        return game_batch_


print_and_log('POSITIVE AFFINE TRANSFORMATIONS',f)
print_and_log(f'Average Distance: {affine_avg_avg_distance:.3f} ({affine_std_avg_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(affine_quantiles_avg_distance)}', f)
print_and_log('', f)

print_and_log('BEST REPLY PRESERVING TRANSFORMATIONS',f)
print_and_log(f'Average Distance: {bestreply_avg_avg_distance:.3f} ({bestreply_std_avg_distance:.3f})', f)
print_and_log(f'Quantiles [0.25, 0.5, 0.75, 0.90, 0.95, 0.99]: {quantiles_string(bestreply_quantiles_avg_distance)}', f)
print_and_log('', f)
'''
