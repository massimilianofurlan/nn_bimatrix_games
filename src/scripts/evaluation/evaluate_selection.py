import argparse
import sys
import torch
import numpy as np
import pandas as pd
from src.utilities.model_utils import *
from src.utilities.data_utils import *
from src.utilities.eval_utils import *
from src.utilities.viz_utils import *
from src.utilities.io_utils import *

parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
parser.add_argument('--model', type=str, default='2x2_default', help="Model folder")
parser.add_argument('--dataset', type=str, default='2x2_default_avon', help="Dataset Folder")
# Process configs
args = parser.parse_args()

# load simulation_metadata
os.system('cls' if os.name == 'nt' else 'clear')
model1, model2, simulation_metadata, model_dir = select_models(model_dir = args.model)
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
eval_file = f'{eval_dir}/evaluation_selection.txt'
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

##################################################################

gamma_distance_nash = evaluation_output['closest_nash_distance']
closest_nash_is_pareto = evaluation_output['closest_nash_is_pareto']
closest_nash_is_utilitarian = evaluation_output['closest_nash_is_utilitarian']
closest_nash_is_harsanyiselten = evaluation_output['closest_nash_is_harsanyiselten']


harsanyiselten_mask = labels['harsanyi_selten_mask']
utilitarian_mask = labels['utilitarian_nash_mask']
payoff_dominant_mask = labels['payoff_dominance_mask']

def create_comparison_table(is_mask1, is_mask2, label1, label2, f):
    # Compute the fractions
    frac_mask1 = np.mean(is_mask1)
    frac_not_mask1 = 1.0 - frac_mask1
    
    frac_mask2 = np.mean(is_mask2)
    frac_not_mask2 = 1.0 - frac_mask2
    
    is_mask2_mask1 = is_mask2 & is_mask1
    frac_mask2_mask1 = np.mean(is_mask2_mask1)
    
    frac_not_mask2_mask1 = frac_mask1 - frac_mask2_mask1
    frac_mask2_not_mask1 = frac_mask2 - frac_mask2_mask1
    
    frac_neither_mask2_nor_mask1 = frac_not_mask2 - frac_not_mask2_mask1
    
    # Define the data
    data = {
        label2: [frac_mask2_mask1, frac_mask2_not_mask1, frac_mask2],
        f'Not {label2}': [frac_not_mask2_mask1, frac_neither_mask2_nor_mask1, frac_not_mask2],
        'Total': [frac_mask1, frac_not_mask1, 1.0]
    }
    
    # Create the DataFrame
    df = pd.DataFrame(data, index=[label1, f'Not {label1}', 'Total'])
    
    # Convert DataFrame to a string
    df_string = f"Comparison Table for '{label1}' and '{label2}':\n{df.to_string()}\n\n"
    
    # Append the DataFrame string to the file
    print_and_log(df_string, file=f)


f = open(eval_file, 'a')

# Example usage
gamma_threshold = 0.1
multiple_nash_mask = statistics['n_nash'] > 1
is_gamma_nash = gamma_distance_nash < gamma_threshold

mask = multiple_nash_mask & is_gamma_nash
print_and_log(f"Multiple Equilibria and 0.1-Nash (applies also to all others) {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Utilitarian", f)

payoff_dominant_exists_mask = statistics['n_payoff_dominant'] == 1
mask = multiple_nash_mask & is_gamma_nash & payoff_dominant_exists_mask
print_and_log(f"When Payoff Dominant exists {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Payoff Dominant", f)



harsanyiselten_is_utilitarian_mask = np.array([np.all(harsanyiselten_mask[z] == utilitarian_mask[z]) for z in range(len(testing_set))])
mask = multiple_nash_mask & is_gamma_nash & harsanyiselten_is_utilitarian_mask
print_and_log(f"When Harsanyi-Selten is Utilitarian {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Payoff Dominant", f)

harsanyiselten_is_not_utilitarian_mask = ~harsanyiselten_is_utilitarian_mask
mask = multiple_nash_mask & is_gamma_nash & harsanyiselten_is_not_utilitarian_mask
print_and_log(f"When Harsanyi-Selten is not Utilitarian {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Payoff Dominant", f)




harsanyiselten_is_payoff_dominant = harsanyiselten_is_utilitarian_mask & payoff_dominant_exists_mask
mask = multiple_nash_mask & is_gamma_nash & harsanyiselten_is_payoff_dominant 
print_and_log(f"When Harsanyi-Selten is Payoff Dominant {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Payoff Dominant", f)

harsanyiselten_is_not_payoff_dominant = harsanyiselten_is_not_utilitarian_mask & payoff_dominant_exists_mask
mask = multiple_nash_mask & is_gamma_nash & harsanyiselten_is_not_payoff_dominant 
print_and_log(f"When Harsanyi-Selten is not Payoff Dominant {np.sum(mask)}", f)
create_comparison_table(closest_nash_is_harsanyiselten[mask], closest_nash_is_utilitarian[mask], 
                        "Harsanyi-Selten", "Payoff Dominant", f)


f.close()
