import argparse
import sys
import os 
import torch
import numpy as np
from src.utilities.model_utils import select_models
from src.utilities.data_utils import select_dataset, load_labels, load_statistics
from src.utilities.eval_utils import evaluate
from src.utilities.viz_utils import plot_cdfs
from src.utilities.io_utils import print_metadata, preview_dataset, log_metadata, save_to_pickle, load_from_pickle, print_and_log

gamma = 0.025

def broadcast(list_arr, func, **kwargs):
    return np.array([func(arr, **kwargs) for arr in list_arr])

def get_mask(n_arr):
    n_arr_mask = [n_arr == k for k in range(0,np.max(n_arr)+1)]
    return n_arr_mask, broadcast(n_arr_mask,np.sum)

def quantiles(arr, quants = np.array([0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1])):
    return np.array2string(np.quantile(arr, quants), formatter={'float_kind': lambda x: f"{x:.3f}"})

def print_evaluation_results(evaluation_output, statistics, mask, f = sys.stdout):
    if ~np.any(mask):
        return
    
    # unpacking input data (evaluation)
    (strategy_profiles, regret_profile, expected_payoff_profile, mass_on_dominated, 
    mass_on_eliminated, gamma_distance_nash, closest_nash_idx, 
    closest_nash_is_pareto, closest_nash_is_utilitarian,
    closest_nash_is_harsanyi_selten, closest_nash_index, dist_from_default) = evaluation_output.values()
    # unpacking input data (game statistics)
    (n_nash, n_pure_nash, n_dominated, n_rationalizable, 
    n_pareto_optimal, n_utilitarian, n_payoff_dominant, n_harsanyi_selten,
    n_index_minus, n_index_zero, n_index_plus) = statistics.values()
    
    # masking input data (evaluation)
    strategy_profiles = strategy_profiles[mask]
    regret_profile = regret_profile[mask]
    expected_payoff_profile = expected_payoff_profile[mask]
    mass_on_dominated = mass_on_dominated[mask]
    mass_on_eliminated = mass_on_eliminated[mask]
    gamma_distance_nash = gamma_distance_nash[mask]
    closest_nash_is_pareto = closest_nash_is_pareto[mask]
    closest_nash_is_utilitarian = closest_nash_is_utilitarian[mask]
    closest_nash_is_harsanyi_selten = closest_nash_is_harsanyi_selten[mask]
    closest_nash_index = closest_nash_index[mask]
    dist_from_default = dist_from_default[mask]
    # masking input data (statistics)
    n_nash = n_nash[mask] 
    n_pure_nash = n_pure_nash[mask] 
    n_mixed_nash = n_nash - n_pure_nash
    n_dominated = n_dominated[mask] 
    n_rationalizable = n_rationalizable[mask] 
    n_pareto_optimal = n_pareto_optimal[mask] 
    n_utilitarian = n_utilitarian[mask] 
    n_payoff_dominant = n_payoff_dominant[mask] 
    n_harsanyi_selten = n_harsanyi_selten[mask] 
    n_index_minus = n_index_minus[mask] 
    n_index_zero = n_index_zero[mask] 
    n_index_plus = n_index_plus[mask] 
    
    # computing submask
    n_nash_masks, n_nash_count = get_mask(n_nash)
    n_pure_nash_masks, n_pure_nash_count = get_mask(n_pure_nash)
    n_mixed_nash_masks, n_mixed_nash_count = get_mask(n_mixed_nash)
    n_pareto_optimal_masks, n_pareto_optimal_count = get_mask(n_pareto_optimal)
    n_utilitarian_masks, n_utilitarian_count = get_mask(n_utilitarian)
    n_payoff_dominant_masks, n_payoff_dominant_count = get_mask(n_payoff_dominant)
    n_harsanyi_selten_masks, n_harsanyi_selten_count = get_mask(n_harsanyi_selten) # almost surely unique for non-degenerate games
    
    
    ################ GAMES STATISTICS ################
    n_games = np.sum(mask)
    share_game = n_games/len(mask)
    freq_nash_idxs = {-1: n_index_minus.sum()/np.sum(n_nash), 
                       0: n_index_zero.sum()/np.sum(n_nash), 
                       1: n_index_plus.sum()/np.sum(n_nash)
    }
    
    ################ INDIVIDUAL RATIONALITY ################
    
    avg_regret = np.mean(regret_profile)
    std_regret = np.std(regret_profile)
    
    # check dominated and eliminated only when there is at least one eliminated
    #has_eliminated = n_rationalizable < n_actions        # player has eliminated
    #n_has_dominated_mask = np.sum(has_eliminated)             # number at least a eliminated 
    
    #if n_has_dominated_mask > 0:
    # gamma-undominated
    avg_gamma_dominated = np.mean(mass_on_dominated)
    std_gamma_dominated = np.std(mass_on_dominated)
    freq_gamma_undominated = np.mean(mass_on_dominated < gamma)
    freq_double_gamma_undominated = np.mean(mass_on_dominated < gamma)
    gamma_dominated_quant = quantiles(mass_on_dominated)
    # gamma-rationalizable (strategy)
    avg_gamma_eliminated = np.mean(mass_on_eliminated)
    std_gamma_eliminated = np.std(mass_on_eliminated)
    freq_gamma_rationalizable = np.mean(mass_on_eliminated < gamma)  
    freq_double_gamma_rationalizable = np.mean(mass_on_eliminated < gamma)  
    gamma_eliminated_quant = quantiles(mass_on_eliminated)
    
    ################ EQUILIBRIUM BEHAVIOR ################
    
    # payoff space distance to nash (regret)
    epsilon_distance_nash = np.max(regret_profile, axis=1)
    avg_epsilon_distance_nash = np.mean(epsilon_distance_nash)
    std_epsilon_distance_nash = np.std(epsilon_distance_nash)
    max_regret_quant = quantiles(epsilon_distance_nash)
    
    # strategy space distance to nash (maximum supremum norm)
    avg_gamma_distance_nash = np.mean(gamma_distance_nash)
    std_gamma_distance_nash = np.std(gamma_distance_nash)
    gamma_distance_nash_quant = quantiles(gamma_distance_nash)
    
    # strategy space distance to rationalizable profiles (mass on eliminated)
    gamma_distance_rationalizable_profile = np.max(mass_on_eliminated, axis=1)
    avg_gamma_distance_rationalizable_profile = np.mean(gamma_distance_rationalizable_profile)
    std_gamma_distance_rationalizable_profile = np.std(gamma_distance_rationalizable_profile)
    gamma_distance_rationalizable_profile_quant = quantiles(gamma_distance_rationalizable_profile)
    
    # frequence gamma-nash
    freq_gamma_nash = np.mean(gamma_distance_nash < gamma)
    freq_double_gamma_nash = np.mean(gamma_distance_nash < gamma*2)
    # frequence gamma-rationalizable (strategy profile)
    freq_gamma_profile_rationalizable =  np.mean(gamma_distance_rationalizable_profile < gamma)
    freq_double_gamma_rationalizable = np.mean(gamma_distance_rationalizable_profile < gamma*2)
    
    ################ SELECTION ################
    
    # equilibrium selection just in case #NE > 1
    #has_multiple_ne = n_nash > 1
    #n_nash_not_singleton = np.sum(has_multiple_ne)
    
    #if n_nash_not_singleton > 0:
    closest_nash_is_pareto = closest_nash_is_pareto
    closest_nash_is_utilitarian = closest_nash_is_utilitarian
    closest_nash_is_harsanyi_selten = closest_nash_is_harsanyi_selten
    closest_nash_is_pareto_harsayani_selten = closest_nash_is_harsanyi_selten & closest_nash_is_pareto
    closest_nash_is_utilitarian_harsayani_selten = closest_nash_is_harsanyi_selten & closest_nash_is_utilitarian
    
    frac_closest_nash_is_pareto = np.mean(closest_nash_is_pareto)
    frac_closest_nash_is_utilitarian = np.mean(closest_nash_is_utilitarian)
    frac_closest_nash_is_harsayani_selten = np.mean(closest_nash_is_harsanyi_selten)
    
    freq_closest_nash_is_pareto_harsayani_selten = np.mean(closest_nash_is_pareto_harsayani_selten)
    freq_closest_nash_is_not_pareto_harsayani_selten = frac_closest_nash_is_harsayani_selten - freq_closest_nash_is_pareto_harsayani_selten
    freq_closest_nash_is_pareto_not_harsayani_selten = frac_closest_nash_is_pareto - freq_closest_nash_is_pareto_harsayani_selten
    
    freq_closest_nash_is_utilitarian_harsayani_selten = np.mean(closest_nash_is_utilitarian_harsayani_selten)
    freq_closest_nash_is_not_utilitarian_harsayani_selten = frac_closest_nash_is_harsayani_selten - freq_closest_nash_is_utilitarian_harsayani_selten
    freq_closest_nash_is_utilitarian_not_harsayani_selten = frac_closest_nash_is_utilitarian - freq_closest_nash_is_utilitarian_harsayani_selten
    
    freq_closest_nash_idx_is_minus = np.mean(closest_nash_index == -1)
    freq_closest_nash_idx_is_zero = np.mean(closest_nash_index == 0)
    freq_closest_nash_idx_is_plus = np.mean(closest_nash_index == 1)
    
    ################ DISTANCE FROM DEFAULT ################
    
    avg_dist_from_default = np.mean(dist_from_default)
    std_dist_from_default = np.std(dist_from_default)
    dist_from_default_quant = quantiles(dist_from_default)
    
    #n_play_mixed = np.any(np.logical_and(strategy_profiles < 0.9, strategy_profiles > 0.1),axis=(1,2)).sum()
    #print(f"Freq. Play Mixed: {n_play_mixed/n_games}")
    ############### OUTPUT TO TERMINAL ################
    
    print_and_log(f'------ Statistics ------', f)
    print_and_log(f'Number of games: {n_games} ({share_game:.3f})', f)
    print_and_log(f'Freq. Number of Nash: {(n_nash_count/n_games).round(3)}', f)
    print_and_log(f'Freq. Number of Pure Nash: {(n_pure_nash_count/n_games).round(3)}', f)
    print_and_log(f'Freq. Number of Mixed Nash: {(n_mixed_nash_count/n_games).round(3)}', f)
    print_and_log(f'Freq. Number of Pareto Superior Nash: {(n_pareto_optimal_count/n_games).round(3)}', f)
    #print_and_log(f'Freq. Number of Utilitarian Nash: {(n_utilitarian_count/n_games).round(3)}', f)
    print_and_log(f'Freq. Number of Payoff Dominant Nash: {(n_payoff_dominant_count/n_games).round(3)}', f)
    #print_and_log(f'Freq. Number of Harsanyi-Selten Nash: {(n_harsanyi_selten_count/n_games).round(3)}', f)
    print_and_log(f'Freq. Nash index (-1,0,1): ({freq_nash_idxs[-1]:.3f}, {freq_nash_idxs[0]:.3f}, {freq_nash_idxs[1]:.3f})', f)
    
    print_and_log(f'------ Equilibrium ------', f)
    print_and_log(f'Average epsilon-Nash: {avg_epsilon_distance_nash:.3f} ({std_epsilon_distance_nash:.3f})', f)
    print_and_log(f'Average gamma-Nash: {avg_gamma_distance_nash:.3f} ({std_gamma_distance_nash:.3f})', f)
    print_and_log(f'Average gamma-Rationalizable Profile: {avg_gamma_distance_rationalizable_profile:.3f} ({std_gamma_distance_rationalizable_profile:.3f})', f)
    print_and_log(f'epsilon-Nash Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {max_regret_quant}', f)
    print_and_log(f'gamma-Nash Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {gamma_distance_nash_quant}', f)
    print_and_log(f'gamma-Rationalizable Profile Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {gamma_distance_rationalizable_profile_quant}', f)
    print_and_log(f'Freq. {gamma}-Nash: {freq_gamma_nash:.3f}', f)
    print_and_log(f'Freq. {2*gamma}-Nash: {freq_double_gamma_nash:.3f}', f)
    print_and_log(f'Freq. {gamma}-Rationalizable Profile: {freq_gamma_profile_rationalizable:.3f}', f)    
    print_and_log(f'Freq. {2*gamma}-Rationalizable Profile: {freq_double_gamma_rationalizable:.3f}', f)
    
    #if n_has_dominated_mask > 0:
    print_and_log(f'------ Individual Rationality ------', f)
    print_and_log(f'Average Regret: {avg_regret:.3f} ({std_regret:.3f})', f)
    print_and_log(f'Average Mass on Dominated: {avg_gamma_dominated:.3f} ({std_gamma_dominated:.3f})', f)
    print_and_log(f'Average Mass on Eliminated: {avg_gamma_eliminated:.3f} ({std_gamma_eliminated:.3f})', f)
    print_and_log(f'Mass on Dominated Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {gamma_dominated_quant}', f)
    print_and_log(f'Mass on Eliminated Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {gamma_eliminated_quant}', f)
    print_and_log(f'Freq. {gamma}-Undominated: {freq_gamma_undominated:.3f}', f)
    print_and_log(f'Freq. {2*gamma}-Undominated: {freq_double_gamma_undominated:.3f}', f)
    print_and_log(f'Freq. {gamma}-Rationalizable: {freq_gamma_rationalizable:.3f}', f)
    print_and_log(f'Freq. {2*gamma}-Rationalizable: {freq_double_gamma_rationalizable:.3f}', f)
    
    #if n_nash_not_singleton > 0:
    print_and_log(f'------ Selection ------', f)
    print_and_log(f'Freq. Closest Nash is Pareto Superior: {frac_closest_nash_is_pareto:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Utilitarian: {frac_closest_nash_is_utilitarian:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Harsanyi-Selten: {frac_closest_nash_is_harsayani_selten:.3f}', f)
    #
    print_and_log(f'Freq. Closest Nash is Pareto Superior & Harsanyi-Selten: {freq_closest_nash_is_pareto_harsayani_selten:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Not Pareto Superior & Harsanyi-Selten: {freq_closest_nash_is_not_pareto_harsayani_selten:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Pareto Superior & Not Harsanyi-Selten: {freq_closest_nash_is_pareto_not_harsayani_selten:.3f}', f)
    #
    print_and_log(f'Freq. Closest Nash is Utilitarian & Harsanyi-Selten: {freq_closest_nash_is_utilitarian_harsayani_selten:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Not Utilitarian & Harsanyi-Selten: {freq_closest_nash_is_not_utilitarian_harsayani_selten:.3f}', f)
    print_and_log(f'Freq. Closest Nash is Utilitarian & Not Harsanyi-Selten: {freq_closest_nash_is_utilitarian_not_harsayani_selten:.3f}', f)
    #
    print_and_log(f'Freq. Closest Nash Index (-1,0,1): ({freq_closest_nash_idx_is_minus:.3f}, {freq_closest_nash_idx_is_zero:.3f}, {freq_closest_nash_idx_is_plus:.3f}) out of ({freq_nash_idxs[-1]:.3f}, {freq_nash_idxs[0]:.3f}, {freq_nash_idxs[1]:.3f})', f)
    
    print_and_log(f'------ Distance from Default ------', f)
    print_and_log(f'Average Distance from nxn_default: {avg_dist_from_default:.3f} ({std_dist_from_default:.3f})', f)
    print_and_log(f'Distance from nxn_default Quantiles (0.25, 0.5, 0.75, 0.90, 0.95, 0.99, 1.0): {dist_from_default_quant}', f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
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
    eval_file = f'{eval_dir}/evaluation.txt'
    log_metadata(eval_file, simulation_metadata, "Models: ")
    log_metadata(eval_file, dataset_metadata, "Dataset: ", 'a')

    f = open(eval_file, 'a')

    print('\nTesting model...')
    evaluation_output = evaluate(model1, model2, testing_set, labels, device)

    print('\nSaving evaluation to file...')
    save_to_pickle(evaluation_output, f'{eval_dir}/evaluation_output.pkl')

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

    regret_profile = evaluation_output['regret_profile']
    mass_on_dominated = evaluation_output['mass_on_dominated']
    mass_on_eliminated = evaluation_output['mass_on_eliminated']
    gamma_distance_nash = evaluation_output['closest_nash_distance']

    #unambiguous_traces = labels['harsanyi_selten_traces_reldiff'] > 0.5

    print_and_log('\n\n[[[ ALL GAMES ]]]', f)
    all_games = np.ones(len(testing_set), dtype=bool)
    print_evaluation_results(evaluation_output, statistics, all_games, f=f)

    print_and_log('\n\n[[[ DOMINANCE SOLVABLE]]]', f)
    dominance_solvable_mask = np.all(statistics['n_rationalizable'] == 1, axis=1)
    print_evaluation_results(evaluation_output, statistics, dominance_solvable_mask, f=f)

    #print_and_log('\n\n[[[ SOME DOMINATED ]]]', f)
    #has_dominated_mask = np.any(statistics['n_dominated'] > 0, axis=1)
    #print_evaluation_results(evaluation_output, statistics, has_dominated_mask, f=f)

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

    print_and_log(f'\n\n[[[ > 1 NASH EQUILIBRIUM & IS {2*gamma}-NASH ]]]', f)
    multiple_nash_mask = statistics['n_nash'] > 1
    gamma_nash_mask = gamma_distance_nash < 2*gamma
    multiple_nash_and_gamma_nash_mask = np.logical_and(multiple_nash_mask, gamma_nash_mask)
    print_evaluation_results(evaluation_output, statistics, multiple_nash_and_gamma_nash_mask, f=f)

    #print_and_log(f'\n\n[[[ 0.9999 QUANTILE GAMMA DISTANCE ]]]', f)
    #high_gamma_distance_nash = gamma_distance_nash >= np.quantile(gamma_distance_nash, 0.9999) 
    #print_evaluation_results(evaluation_output, statistics, high_gamma_distance_nash, f=f)

    #print_and_log(f'\n\n[[[ 0.9999 QUANTILE EPSILON DISTANCE ]]]', f)
    #epsilon_distance_nash = np.max(regret_profile,axis=1)
    #high_epsilon_distance_nash = epsilon_distance_nash >= np.quantile(epsilon_distance_nash, 0.9999) 
    #print_evaluation_results(evaluation_output, statistics, high_epsilon_distance_nash, f=f)

    #for k in range(1,max(statistics['n_nash'])+1):
    #    has_k_ne = statistics['n_nash'] == k
    #    if sum(has_k_ne) != 0:
    #        print_and_log(f'\n\n[[[ {k} NASH EQUILIBRIA ]]]', f)
    #        print_evaluation_results(evaluation_output, statistics, has_k_ne, f=f)

    v_norm_A, v_norm_B = simulation_metadata['training_set']['normal_vectors']
    if v_norm_A or v_norm_B:
        A_vec = testing_set[:,0,:,:].transpose(0,2,1).reshape(-1, dataset_metadata['n_actions']**2)
        B_vec = testing_set[:,1,:,:].reshape(-1, dataset_metadata['n_actions']**2)
        A_inners = np.matmul(A_vec, np.array(v_norm_A)) if v_norm_A else np.ones(A_vec.shape[0])
        B_inners = np.matmul(B_vec, np.array(v_norm_B)) if v_norm_B else np.ones(B_vec.shape[0])
        subspace_mask = np.logical_and(A_inners>0, B_inners>0)

        print_and_log('\n\n[[SUBSPACE]]', f)
        print_evaluation_results(evaluation_output, statistics, subspace_mask, f=f)

        print_and_log('\n\n[[SUBSPACE - 0 PURE NASH EQUILIBRIA ]]', f)
        subspace_zero_pure_nash_mask = np.logical_and(subspace_mask,zero_pure_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_zero_pure_nash_mask, f=f)

        print_and_log('\n\n[[SUBSPACE - 1 PURE NASH EQUILIBRIA ]]', f)
        subspace_one_pure_nash_mask = np.logical_and(subspace_mask,one_pure_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_one_pure_nash_mask, f=f)

        print_and_log('\n\n[[SUBSPACE - >1 PURE NASH EQUILIBRIA ]]', f)
        subspace_multiple_pure_nash_mask = np.logical_and(subspace_mask,multiple_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_multiple_pure_nash_mask, f=f)
    
        print_and_log('\n\n[[COMPLEMENT OF SUBSPACE]]', f)
        subspace_c_mask = ~subspace_mask
        print_evaluation_results(evaluation_output, statistics, subspace_c_mask, f=f)

        print_and_log('\n\n[[COMPLEMENT SUBSPACE - 0 PURE NASH EQUILIBRIA ]]', f)
        subspace_c_zero_pure_nash_mask = np.logical_and(subspace_c_mask,zero_pure_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_c_zero_pure_nash_mask, f=f)

        print_and_log('\n\n[[COMPLEMENT SUBSPACE - 1 PURE NASH EQUILIBRIA ]]', f)
        subspace_c_one_pure_nash_mask = np.logical_and(subspace_c_mask,one_pure_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_c_one_pure_nash_mask, f=f)
    
        print_and_log('\n\n[[COMPLEMENT SUBSPACE - >1 PURE NASH EQUILIBRIA ]]', f)
        subspace_c_multiple_pure_nash_mask = np.logical_and(subspace_c_mask,multiple_nash_mask)
        print_evaluation_results(evaluation_output, statistics, subspace_c_multiple_pure_nash_mask, f=f)
    
    epsilon_distance_nash = np.max(regret_profile, axis=1)
    plot_cdfs(epsilon_distance_nash, epsilon_distance_nash[zero_pure_nash_mask], 
          eval_dir, file_name="regret_cdf.pdf", 
          xlabel='MaxReg', ylabel='eCDF',
          label1='All Games', label2='No Pure Nash', 
          title = rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}')
    plot_cdfs(gamma_distance_nash, gamma_distance_nash[zero_pure_nash_mask], 
          eval_dir, file_name="gamma_cdf.pdf", 
          xlabel='DistNash', ylabel='eCDF', 
          label1='All Games', label2='No Pure Nash', 
          title = rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}')
    plot_cdfs(mass_on_dominated[dominance_solvable_mask].flatten(), 
              mass_on_eliminated[dominance_solvable_mask].flatten(), 
              eval_dir, file_name="dominance_cdf.pdf", 
              xlabel='Probability Mass', ylabel='eCDF',
              label1='Dominated', label2='Eliminated', ylim_left=0.92, 
              title = rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}')

    f.close()


if __name__ == "__main__":
    main()