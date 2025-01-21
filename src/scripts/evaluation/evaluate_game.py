import argparse
import sys
import torch
import numpy as np
import math
import itertools

from src.utilities.model_utils import *
from src.utilities.data_utils import *
from src.utilities.eval_utils import *
from src.utilities.bimatrix_utils import *
from src.utilities.viz_utils import *
from src.utilities.io_utils import *
from src.modules.loss_function import Loss


class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    DARK_GREEN = '\033[32m'
    YELLOW = '\033[93m'
    PURPLE = '\033[94m'
    BLUE = "\033[0;34m"
    DEFAULT = "\033[0m"
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    ORANGE = "\033[38;5;214m"
    END = '\033[0m'


##################################################################

def transform_batch_games(games_batch, n_actions):
    # Demean games x <- x - mean(x)
    games_batch -= torch.mean(games_batch, dim=(2, 3), keepdim=True)
    # Rescale games into unit sfere x <- x / norm(x)
    games_batch /=  torch.linalg.matrix_norm(games_batch, dim=(2, 3), keepdim=True)
    # Unit variance (sfere of radius n_actions) x <-  x <- x * n_actions
    games_batch *= n_actions
    return games_batch

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
    parser.add_argument('--model', type=str, default='2x2_default', help="Model folder")
    # Process configs
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # load model_metadata
    os.system('cls' if os.name == 'nt' else 'clear')
    model1, model2, model_metadata, model_dir = select_models(model_dir = args.model, device=device)
    model1.eval()
    model2.eval()

    # visualize to terminal
    print(f"\nModel: ")
    print_metadata(model_metadata)


    # all done
    print(f"All done")

    n_actions_model = model_metadata['n_actions']
    games = read_games_from_file("games.txt")
    n_traces = 10000

    with torch.no_grad():
        for game_name, game in games.items():    

            # convert game to numpy and tensor
            game_np32 = np.array(game, dtype=np.float32)
            game_np64 = np.array(game, dtype=np.float64)
            game_tensor = torch.tensor(game_np32, device=device)
            _, n_actions, _ = game_np64.shape

            if n_actions > n_actions_model:
                continue

            # show game
            print(f"\nGame: {Color.PURPLE}{game_name}{Color.END}")
            print_bimatrix_game(*game)

            # analysis

            # mask dominated strategy
            dominated_mask1 = get_dominated_mask(game_np64[0])
            dominated_mask2 = get_dominated_mask(game_np64[1])
            # mask rationalizable strategy profiles
            rationalizable_mask = get_rationalizable_mask(game_np64)
            # compute nash equilibria
            set_nash, set_nash_payoffs = get_nash_equilibria(game_np64, rational=False)
            # mask pure nash equilibria 
            pure_nash_mask = get_pure_nash_mask(set_nash)
            # mask pareto optimal and utilitarian nash equilibria
            pareto_nash_mask, utilitarian_nash_mask, payoff_dominance_mask = get_pareto_optimal_nash_mask(game_np64, set_nash_payoffs)
            # mask harsanyi-selten linear tracing selected nashe quilibria
            harsanyi_selten_mask, harsanyi_selten_traces, harsanyi_selten_traces_reldiff = get_harsanyi_selten_mask(game_np64, set_nash, n_trace=n_traces)
            # compute stability index of each nash equilibrium
            nash_index = get_indeces(game_np64, set_nash)
            # compute minimax strategies and payoffs
            maxmin_payoffs = get_maxmin_payoff(game_np64)

            # transform game and feed it to network

            # add batch dimension
            G = game_tensor.unsqueeze(0)
            # reshape to n_actions x n_actions
            G = border_game(G, n_actions, n_actions_model, max_payoff=n_actions ** 2)
            # transpose
            G_transpose = transpose_game(G)

            # get actions
            #G_transformed = transform_batch_games(G.clone(),n_actions)
            #G_transpose_transformed = transform_batch_games(G_transpose.clone(),n_actions)
            p = model1(G)
            q = model2(G_transpose)

            # remove border to original game
            G_unbord = G[:, :, :n_actions, :n_actions]
            G_transpose_unbord = transpose_game(G_unbord)
            p_unbord = p[:, :n_actions]
            q_unbord = q[:, :n_actions]

            expected_payoff1 = get_expected_payoff(G_unbord, p_unbord, q_unbord)
            expected_payoff2 = get_expected_payoff(G_transpose_unbord, q_unbord, p_unbord)
            regret1 = Loss.regret(G_unbord, p_unbord, q_unbord)
            regret2 = Loss.regret(G_transpose_unbord, q_unbord, p_unbord)

            # find closest nash for each transformation
            strategy_profile_unbord = torch.stack([p_unbord, q_unbord], dim=1)
            closest_nash_idx, closest_nash_distance = get_closest_nash(strategy_profile_unbord, [set_nash])

            close_to_nash_mask = np.array(closest_nash_distance) < 0.1
            nash_masks = np.array([(closest_nash_idx == k) & close_to_nash_mask for k in range(len(set_nash))])
            nash_count = np.array([np.sum(mask) for mask in nash_masks])
            nash_sort_idxs = np.argsort(-nash_count)

            # Print outcomes
            p = p.detach().cpu().numpy().flatten()
            q = q.detach().cpu().numpy().flatten()
            action1_formatted = '[' + ', '.join([f"{x:.2f}".rstrip('0') + ' ' * (4 - len(str(x).rstrip('0'))) for x in p.round(2)]) + ']'
            action2_formatted = '[' + ', '.join([f"{x:.2f}".rstrip('0') + ' ' * (4 - len(str(x).rstrip('0'))) for x in q.round(2)]) + ']'
            print(f"\nModel: {Color.RED}{action1_formatted}{Color.END}, {Color.RED}{action2_formatted}{Color.END}")#, end="  ")
            #print(f"EP: {Color.RED}[" + " ".join([f"{('+' if x >= 0 else '')}{x:.2f}" for x in expected_payoff]) + f"]{Color.END}")
            # Print Nash equilibria
            for i, (ne_action1, ne_action2) in enumerate(set_nash):
                # Round each element and format it with three decimals
                ne_action1_formatted = '[' + ', '.join([f"{x:.2f}".rstrip('0') + ' ' * (4 - len(str(x).rstrip('0'))) for x in ne_action1.round(2)]) + ']'
                ne_action2_formatted = '[' + ', '.join([f"{x:.2f}".rstrip('0') + ' ' * (4 - len(str(x).rstrip('0'))) for x in ne_action2.round(2)]) + ']'
                #if i == closest_nash_idx & len(set_nash) > 1:
                #    print(f" NE {i+1}: {Color.ORANGE}{ne_action1_formatted}{Color.END}, {Color.ORANGE}{ne_action2_formatted}{Color.END}", end="  ")
                #else:
                print(f" NE {i+1}: {ne_action1_formatted}, {ne_action2_formatted}", end="  ")
                #print("EP: [" + " ".join([f"{('+' if x >= 0 else '')}{x:.2f}" for x in expected_payoff_ne[i]]) + "]", end="  ")
                print(f"CH: [{nash_index[i]}] ",end="")

                print(" [**]" if utilitarian_nash_mask[i] else " [*]" if pareto_nash_mask[i] else "", end="")
                print(f" [HS] [{harsanyi_selten_traces[i]}]" if harsanyi_selten_mask[i] else f" [{harsanyi_selten_traces[i]}]")
            # Print game summary
            print(f"Regrets: {Color.DEFAULT}[{regret1.item():.4f}{Color.END}, {Color.DEFAULT}{regret2.item():.4f}]{Color.END}")
            print(f"Variation Distance: {Color.DEFAULT}{closest_nash_distance.item():.2f}{Color.END}")
            print(f"Dominated actions:  {dominated_mask1.astype(int)}, {dominated_mask2.astype(int)}")
            print(f"Rationalizable set: {rationalizable_mask[0].flatten().astype(int)}, {rationalizable_mask[1].flatten().astype(int)}")
            print("\n-------------------------------------------")
