import time
import numpy as np
import torch
from src.utilities.training_utils import transpose_game
from src.modules.loss_function import Loss

def get_expected_payoff(G,x,y):
    # return expected payoff of row player of G
    A = G[:, 0, :, :]
    # vector of expected payoffs from each pure strategy 
    Ay = torch.bmm(A, y.unsqueeze(2)).squeeze(2)  # shape: batch_size x n_actions
    # expected payoff from strategy x
    xAy = torch.bmm(x.unsqueeze(1), Ay.unsqueeze(2)).squeeze()  # shape: batch_size
    return xAy

def get_masked_probability(mixed_strategy: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # calculate total probability mass on masked mixed_strategy across batch dimension
    return torch.sum(mixed_strategy * mask.float(), dim=1)

def get_closest_nash(strategy_profiles: torch.Tensor, set_nash):
    # find the closest Nash equilibrium and compute the distances for each game in the batch.
    device = strategy_profiles.device
    max_n_nash = torch.max(torch.tensor([arr.shape[0] for arr in set_nash], device=device))
    # expand strategy_profile for broadcasting
    strategy_profile_expanded = strategy_profiles.unsqueeze(1).expand(-1, max_n_nash, -1, -1)
    # stack set_nash tensors and pad them
    set_nash_tensor = [torch.tensor(np.float32(arr), dtype=torch.float32, device=device) for arr in set_nash]
    set_nash_expanded = torch.nn.utils.rnn.pad_sequence(set_nash_tensor, batch_first=True, padding_value=float('inf'))
    # compute distances (maximum total variation across agents)
    dists = torch.amax(torch.sum(torch.abs(set_nash_expanded - strategy_profile_expanded), dim=3), dim=2) * 0.5
    # compute distances (maximum supremum norm across agents)
    #dists = torch.amax(torch.abs(set_nash_expanded - strategy_profile_expanded), dim=(2, 3))
    min_dist, argmin_dist = dists.min(dim=1)
    return argmin_dist.detach().cpu().numpy(), min_dist.detach().cpu().numpy()

def get_value(indices: np.ndarray, array_list: np.ndarray) -> np.ndarray:
    # get value from each array in a list of arrays based on provided indices.
    values = np.zeros(len(indices), dtype=bool)
    for i, idx in enumerate(indices):
        values[i] = array_list[i][idx]
    return values

def border_game(game_batch, n_actions, n_actions_model, max_payoff = None):
    # border n_actions x n_actions game with a strictly dominated strategy 
    # to match n_actions_model x n_actions_model 
    # action in position 0 strictly dominates action on borders 
    if n_actions == n_actions_model:
        return game_batch
    else:
        max_payoff = (n_actions**2-1)**0.5 if max_payoff == None else max_payoff
        device = game_batch.device
        batch_size = game_batch.shape[0]
        game_batch_ = torch.zeros(batch_size, 2, n_actions_model, n_actions_model, dtype=torch.float32, device = device)
        game_batch_[:, :, 0:n_actions, 0:n_actions] = game_batch
        game_batch_[:, 0, n_actions:, :] = game_batch_[:, 0, 0, :].unsqueeze(1) - max_payoff/4 
        game_batch_[:, 1, :, n_actions:] = game_batch_[:, 1, :, 0].unsqueeze(2) - max_payoff/4
        return game_batch_

def evaluate(model1: torch.nn.Module, model2: torch.nn.Module, testing_set: np.ndarray, labels: dict, device: torch.device, batch_size: int = 16384):
    """
    Evaluate self-play performance of a model on a batch of games.

    Args:
        model (torch.nn.Module): Model to evaluate.
        testing_set (np.ndarray): Batch of games to evaluate on.
        labels (Dict[str, np.ndarray]): Dictionary containing labels for the games.
        device (torch.device): Device to perform evaluation on.
        batch_size (int): Batch size for evaluation.

    Returns:
        Tuple: Results of the evaluation.
    """
    # convert testing_set to tensor
    testing_set = np.array(testing_set, dtype=np.float32)
    testing_set = torch.tensor(testing_set, device=device, dtype=torch.float32, requires_grad=False)

    n_games, n_players, n_actions, _ = testing_set.size() # n_actions may differ from model input size
    testing_set = border_game(testing_set, n_actions, model1.n_actions)

    # Process labels
    set_nash = labels['set_nash']
    dominated_mask = torch.from_numpy(np.array(labels['dominated_mask'])).bool().to(device)
    rationalizable_mask = torch.from_numpy(np.array(labels['rationalizable_mask'])).bool().to(device)
    pareto_nash_mask = labels['pareto_nash_mask']
    utilitarian_nash_mask = labels['utilitarian_nash_mask']
    #payoff_dominant_mask = labels['payoff_dominance_mask']
    harsanyi_selten_mask = labels['harsanyi_selten_mask']
    nash_index = labels['nash_index']

    strategy_profiles = torch.empty(n_games, n_players, n_actions, device=device, dtype=torch.float16)
    regret_profile = torch.empty(n_games, n_players, device=device, dtype=torch.float16)
    expected_payoff_profile = torch.empty(n_games, n_players, device=device)
    mass_on_dominated = torch.empty(n_games, n_players, device=device, dtype=torch.float16)
    mass_on_eliminated = torch.empty(n_games, n_players, device=device, dtype=torch.float16)
    closest_nash_distance = np.empty(n_games, dtype=np.float16)
    closest_nash_idx = np.empty(n_games, dtype=np.int8)
    closest_nash_is_pareto = np.empty(n_games, dtype=bool)
    closest_nash_is_utilitarian = np.empty(n_games, dtype=bool)
    #closest_nash_is_payoff_dominant = np.empty(n_games, dtype=bool)
    closest_nash_is_harsanyiselten = np.empty(n_games, dtype=bool)
    closest_nash_stability_index = np.empty(n_games, dtype=np.int8)

    with torch.no_grad():  # Disable gradient computation during evaluation
        start_time = time.time()
        for start_index in range(0, n_games, batch_size):
            # Process input batch
            end_index = start_index + min(batch_size, n_games - start_index)
            G = testing_set[start_index:end_index]
            G_transpose = transpose_game(G)

            # Forward pass
            p = model1(G)
            q = model2(G_transpose)

            # sample from uniform distribution on n_actions-simplex: x/|x| with x~Exp(1)
            #p = -torch.log(torch.rand((batch_size,n_actions), dtype=torch.float32, device='mps'))
            #p /= p.sum(axis=1, keepdim=True)
            #q = -torch.log(torch.rand((batch_size,n_actions), dtype=torch.float32, device='mps'))
            #q /= q.sum(axis=1, keepdim=True)
            # constant uniform
            #p = torch.ones((batch_size,n_actions), dtype=torch.float32, device='mps')/n_actions
            #q = torch.ones((batch_size,n_actions), dtype=torch.float32, device='mps')/n_actions

            # compute regrets
            regret_profile[start_index:end_index, 0] = Loss.regret(G, p, q)
            regret_profile[start_index:end_index, 1] = Loss.regret(G_transpose, q, p)
            
            # compute expected payoffs 
            expected_payoff_profile[start_index:end_index,0] = get_expected_payoff(G,p,q)
            expected_payoff_profile[start_index:end_index,1] = get_expected_payoff(G_transpose,q,p)

            p = p[:,:n_actions]
            q = q[:,:n_actions]

            # Log strategy profiles
            strategy_profiles[start_index:end_index, 0, :] = p
            strategy_profiles[start_index:end_index, 1, :] = q

            # Check if agents play dominated strategy with prob higher than 0.05
            mass_on_dominated[start_index:end_index, 0] = get_masked_probability(p, dominated_mask[start_index:end_index, 0, :])
            mass_on_dominated[start_index:end_index, 1] = get_masked_probability(q, dominated_mask[start_index:end_index, 1, :])
            mass_on_eliminated[start_index:end_index, 0] = get_masked_probability(p, ~rationalizable_mask[start_index:end_index, 0, :])
            mass_on_eliminated[start_index:end_index, 1] = get_masked_probability(q, ~rationalizable_mask[start_index:end_index, 1, :])

            closest_nash_idx_, closest_nash_distance_ = get_closest_nash(strategy_profiles[start_index:end_index, :, :], set_nash[start_index:end_index])
            closest_nash_idx[start_index:end_index] = closest_nash_idx_
            closest_nash_distance[start_index:end_index] = closest_nash_distance_

            closest_nash_is_pareto[start_index:end_index] = get_value(closest_nash_idx_, pareto_nash_mask[start_index:end_index])
            closest_nash_is_utilitarian[start_index:end_index] = get_value(closest_nash_idx_, utilitarian_nash_mask[start_index:end_index])
            #closest_nash_is_payoff_dominant[start_index:end_index] = get_value(closest_nash_idx_, payoff_dominant_mask[start_index:end_index])
            closest_nash_is_harsanyiselten[start_index:end_index] = get_value(closest_nash_idx_, harsanyi_selten_mask[start_index:end_index])
            closest_nash_stability_index[start_index:end_index] = get_value(closest_nash_idx_, nash_index[start_index:end_index])

            progress_percentage = (end_index / n_games) * 100
            print(f"\rProgress: {progress_percentage:.2f}%, Time Elapsed: {time.time() - start_time:.0f} sec", end='', flush=True)

    print("\nEvaluation complete.")

    # Convert tensors to numpy arrays for serialization
    evaluation_output = {
        'strategy_profiles': strategy_profiles.cpu().numpy(),
        'regret_profile': regret_profile.cpu().numpy(),
        'expected_payoff_profile': expected_payoff_profile.cpu().numpy(),
        'mass_on_dominated': mass_on_dominated.cpu().numpy(),
        'mass_on_eliminated': mass_on_eliminated.cpu().numpy(),
        'closest_nash_distance': closest_nash_distance,
        'closest_nash_idx': closest_nash_idx,
        'closest_nash_is_pareto': closest_nash_is_pareto,
        'closest_nash_is_utilitarian': closest_nash_is_utilitarian,
        #'closest_nash_is_payoff_dominant': closest_nash_is_payoff_dominant,
        'closest_nash_is_harsanyiselten': closest_nash_is_harsanyiselten,
        'closest_nash_stability_index': closest_nash_stability_index
    }

    return evaluation_output
