import argparse
import sys
import torch
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
_, _, simulation_metadata, model_dir = select_models(model_dir = args.model)

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

# iterate over model steps
n_optim_steps = simulation_metadata['optimization']['optimization_steps']
exp = np.ceil(np.log10(n_optim_steps)).astype(int)
model_log_steps = np.unique(np.logspace(0, exp, num=20*exp+1, dtype=int))
model_log_steps = model_log_steps[model_log_steps <= n_optim_steps]
evaluation_outputs = []
for step in model_log_steps:
    # Load models
    model1 = load_model(simulation_metadata['model1'], f'models/{model_dir}/models_log/model1_{step}.pth', device).eval()
    model2 = load_model(simulation_metadata['model2'], f'models/{model_dir}/models_log/model2_{step}.pth', device).eval()
    evaluation_output = evaluate(model1, model2, testing_set, labels, device)
    evaluation_outputs.append(evaluation_output)
    print(f'Step {step}\n')

final_evaluation = load_from_pickle(f'models/{model_dir}/{dataset_dir}/evaluation_output.pkl')
evaluation_outputs.append(final_evaluation)
model_log_steps = np.append(model_log_steps,n_optim_steps)
# SAVE WHEN COMPUTING
save_to_pickle(evaluation_outputs, f'models/{model_dir}/{dataset_dir}/learning_outputs.pkl')

# LOAD IF ALREADY COMPUTED
#evaluation_outputs = load_from_pickle(f'models/{model_dir}/{dataset_dir}/learning_outputs.pkl')
#model_log_steps = np.append(model_log_steps,n_optim_steps)

regret_profiles = np.array([evaluation_outputs[step]['regret_profile'] for step in range(len(evaluation_outputs))], dtype=np.float16)
closest_nash_distance = np.array([evaluation_outputs[step]['closest_nash_distance'] for step in range(len(evaluation_outputs))], dtype=np.float16)

n_actions = simulation_metadata['model1']['n_actions']
plot_learning_curves(regret_profiles.max(axis=2), statistics, model_log_steps, 
                     base_dir = f'models/{model_dir}/{dataset_dir}', file_name="avg_maxreg.pdf", 
                     xlabel='Step', ylabel='Avg. MaxReg', 
                     title=rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}',
                     legend_labels=[r'Some Pure Nash ', r'No Pure Nash'],
                     confidence = 0)

plot_learning_curves(closest_nash_distance, statistics, model_log_steps, 
                     base_dir = f'models/{model_dir}/{dataset_dir}', file_name="avg_gamma.pdf", 
                     xlabel='Step', ylabel='Avg. MaxDistNash', 
                     title=rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}',
                     legend_labels=[r'Some Pure Nash', r'No Pure Nash'],
                     confidence = 0)

# fits 
if n_actions == 2:
    exp_fit_range = [75, 140]
    power_fit_range = [140, None]
elif n_actions == 3:
    exp_fit_range = [200, 600]
    power_fit_range = [600, None]

plot_learning_curve_fit(regret_profiles, model_log_steps,
                        base_dir=f'models/{model_dir}/{dataset_dir}', file_name=f"curvefits.pdf",
                        xlabel='Step', ylabel='Avg. MaxReg',
                        title=rf'$\mathbf{{{n_actions} \times {n_actions}}}$ \textbf{{Games}}',
                        exp_fit_range=exp_fit_range, power_fit_range=power_fit_range)

