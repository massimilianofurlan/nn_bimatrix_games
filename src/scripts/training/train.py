import argparse
import torch
import os
from src.utilities.io_utils import clear, read_config, print_metadata, save_to_pickle
from src.utilities.model_utils import (save_model, initialize_model, initialize_optimizer, 
                                        initialize_scheduler, generate_metadata, initialize_weigths)
from src.utilities.viz_utils import plot_training_loss
from src.utilities.data_utils import load_dataset
from src.utilities.training_utils import train
from src.modules.loss_function import Loss
from src.modules.sampler import BimatrixSampler

def resolve_conflicts_trainingset(args, config):
    # overwrite configs to trainingst configs
    training_set =  torch.tensor(load_dataset(args.training_set), dtype=torch.float)
    config['n_actions'] = training_set.size(2)
    config['payoffs_space'] = 'set:' + args.training_set
    config['game_class'] = 'set:' + args.training_set
    return config, training_set

def resolve_conflicts_model(args, config):
    # overwrite configs to initial model configs
    with open(f'models/{args.init_model}/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    # overwrite config
    config['payoffs_space'] = model_metadata['payoffs_space']
    config['game_class'] = model_metadata['game_class']
    config['model_class'] = model_metadata['model_class']
    config['n_actions'] = model_metadata['n_actions']
    config['n_layers'] = model_metadata['n_layers']
    config['hidden_dim'] = model_metadata['hidden_dim']
    config['ex_ante'] = model_metadata['ex_ante']
    config['p'] = model_metadata['p']
    # overwrite args
    args.n_games = model_metadata['n_games']
    args.batch_size = model_metadata['batch_size']
    args.optimizer = model_metadata['optimizer']
    args.gamma = model_metadata['gamma']
    args.lr *= args.gamma**model_metadata['optimization_steps']
    return args, config

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--training_set', type=str, default=None, help="Training set (overrides config.toml)")
    parser.add_argument('--n_games', type=int, default=2**25, help="Number of games to train on")
    parser.add_argument('--config', type=str, default="2x2_default")    
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer (Adam or SGD)")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=1, help="Decay rate for LR scheduler")
    parser.add_argument('--init_model', type=str, default=None, help="Pre-trained model")    
    parser.add_argument('--log_models',  action='store_true', help="Log models")    
    parser.add_argument('--name', type=str, default=None, help="Model name")
    parser.add_argument('--seed', type=int, default=1, help="Seed")
 
    # Process configs
    args = parser.parse_args()
    config = read_config('config.toml', args.config)
    (args, config) = (args, config) if not args.init_model else resolve_conflicts_model(args, config)
    (config, training_set) = (config, None) if not args.training_set else resolve_conflicts_trainingset(args, config)

     # Setting seed
    torch.manual_seed(args.seed)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Initialize row player network
    model1 = initialize_model(config, device)
    optimizer1 = initialize_optimizer(model1, args.optimizer, args.lr)
    scheduler1 = initialize_scheduler(optimizer1, args.gamma)
   
    # Initialize column player network
    model2 = initialize_model(config, device)
    optimizer2 = initialize_optimizer(model2, args.optimizer, args.lr)
    scheduler2 = initialize_scheduler(optimizer2, args.gamma)
   
    # Initialize weights to initial models weigths if necessary
    (model1, model2) = (model1, model2) if not args.init_model else initialize_weigths(model1, model2, args.init_model)

    # Loss function
    loss_function = Loss(ex_ante = config['ex_ante'], p = config['p'])
    
    # Bimatrix sampler
    rand_bimatrix = BimatrixSampler(n_actions=config['n_actions'], payoffs_space=config['payoffs_space'], 
                                    game_class=config["game_class"], set_games = training_set, device=device)
    # if ex_post loss repeat sampled games
    rand_bimatrix_ = lambda batch_size: rand_bimatrix(batch_size).repeat(64, 1, 1, 1).view(64 * batch_size, 2, config['n_actions'], config['n_actions'])
    rand_bimatrix_ = rand_bimatrix if config['ex_ante'] else rand_bimatrix_

    # Generate metadata
    model_metadata, model_timestamp = generate_metadata(config, args, model1)
    print("\nModel Metadata:")
    print_metadata(model_metadata)

    # Train the model
    print("\nStarting training...\n")
    timestamp = model_timestamp * args.log_models
    model1, model2, avg_regrets = train(model1, optimizer1, scheduler1, model2, optimizer2, scheduler2,
                                        args.n_games, args.batch_size, loss_function, rand_bimatrix_, timestamp)

    # Save trained model
    model_path = os.path.join("models", args.name if args.name else model_timestamp)
    save_model(model1, model_path, file_name="model1.pth", metadata=model_metadata, verbose=True)
    save_model(model2, model_path, file_name="model2.pth", verbose=True)

    # Save regrets and grad norms
    filename = f'avg_regrets_{args.init_model}.pkl' if args.init_model else 'avg_regrets.pkl'
    save_to_pickle(avg_regrets, os.path.join(model_path, filename))

    # Plot training loss
    n_actions = config['n_actions']
    plot_training_loss(avg_regrets, model_path, file_name="learning_curve.pdf", xlabel='Step', ylabel='MaxReg', 
                       title=rf'$\mathbf{{{n_actions}}} \times \mathbf{{{n_actions}}}$ Games')

if __name__ == "__main__":
    clear()
    main()

