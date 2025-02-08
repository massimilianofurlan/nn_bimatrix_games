import os
import json
import torch
import datetime
import copy
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from src.utilities.io_utils import display_info, select_item
from src.modules.mlp import MLP_Bimatrix

def initialize_model(config, device):
    # initialize model based on config
    model_class = config['model_class']
    nn_config = {key: config[key] for key in ['n_actions', 'hidden_dim', 'n_layers']}
    if model_class == "mlp":
        model = MLP_Bimatrix(**nn_config).to(device)
    model.device = device
    model.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config['device'] = str(model.device)
    config['n_params'] = model.n_params
    return model

def initialize_optimizer(model, optim_algorithm, lr):
    # initialize and return the optimizer
    if optim_algorithm == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optim_algorithm == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer

def initialize_scheduler(optimizer, gamma):
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    return scheduler


def generate_metadata(config, args):
    # generate metadata
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    bimatrix_metadata = {
        'payoffs_space': config['bimatrix']['payoffs_space'],
        'game_class': config['bimatrix']['game_class'],
        'normal_vectors': config['bimatrix']['normal_vectors'],
    }
    model1_metadata = {
        'model_class': config['model1']['model_class'],
        'n_layers': config['model1']['n_layers'],
        'hidden_dim': config['model1']['hidden_dim'],
        'n_params': config['model1']['n_params'],
    }
    model2_metadata = {
        'model_class': config['model2']['model_class'],
        'n_layers': config['model2']['n_layers'],
        'hidden_dim': config['model2']['hidden_dim'],
        'n_params': config['model2']['n_params'],
    }
    optim_metadata = {
        'learning_rate': args.lr,
        'gamma': args.gamma,
        'optimization_steps': args.n_games // args.batch_size,
        'optimizer': args.optimizer,
    }
    loss_metadata = {
        'ex_ante': config['loss']['ex_ante'],
        'p': config['loss']['p'],
    }
    metadata = {
        'timestamp': timestamp,
        'n_actions': config['n_actions'],
        **bimatrix_metadata,
        'model1': model1_metadata, 
        'model2': model2_metadata,
        'n_games': args.n_games,
        'batch_size': args.batch_size,
        **optim_metadata,
        **loss_metadata,
        'seed': args.seed,
    }
    return metadata, timestamp

def initialize_weigths(model1, model2, init_model):
    device = model1.device
    model1_path = os.path.join('models', init_model, "model1.pth")
    model2_path = os.path.join('models', init_model, "model2.pth")
    model1_weigths = torch.load(model1_path, map_location=torch.device(device), weights_only=True)['model_state_dict']
    model2_weigths = torch.load(model2_path, map_location=torch.device(device), weights_only=True)['model_state_dict']
    model1.load_state_dict(model1_weigths)
    model2.load_state_dict(model2_weigths)
    return model1, model2


def select_models(model_dir = None, device='cpu'):
    """Allow user to select a model and load its testing set."""
    if not model_dir:
        model_info = display_info('models', 'simulation_metadata.json')
        model_dir, _ = select_item(model_info)
    with open(f'models/{model_dir}/simulation_metadata.json', 'r') as f:
        simulation_metadata = json.load(f)
    model1 = load_model(simulation_metadata, f'models/{model_dir}/model1.pth', device)
    model2 = load_model(simulation_metadata, f'models/{model_dir}/model2.pth', device)
    return  model1, model2, simulation_metadata, model_dir

def load_model(simulation_metadata, model_path, device='cpu'):
    """Load a specific model and its metadata from a model directory."""
    if simulation_metadata['model_class'] == "mlp":
        model = MLP_Bimatrix(n_actions=simulation_metadata['n_actions'],
                             hidden_dim=simulation_metadata['hidden_dim'],
                             n_layers=simulation_metadata['n_layers']).to(device)
    # Load the model weights
    model_weigths = torch.load(model_path, map_location=torch.device(device), weights_only=True)['model_state_dict']
    model.load_state_dict(model_weigths)
    return model

def save_model(model, base_dir, file_name='model.pth', metadata=None, verbose=False):
    """Save model (and optionally metadata) to base_dir"""

    # Create the base directory for models if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Define the path for the model and metadata files
    path = os.path.join(base_dir, file_name)
    
    # Save the model
    torch.save({'model_state_dict': model.state_dict()}, path)
    
    # Save model metadata if provided
    if metadata is not None:
        metadata_file_name = os.path.join(base_dir, "simulation_metadata.json")
        with open(metadata_file_name, 'w') as f:
            json.dump(metadata, f, indent=4)

    if verbose:
        print(f"Model saved in {path}")

    return base_dir