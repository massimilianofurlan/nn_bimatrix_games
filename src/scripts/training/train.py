import argparse
import torch
import os
from src.utilities.io_utils import clear, read_config, print_metadata, save_to_pickle, process_config, save_metadata
from src.utilities.model_utils import (save_model, initialize_model, initialize_optimizer, 
                                        initialize_scheduler, generate_metadata, initialize_weigths)
from src.utilities.viz_utils import plot_training_loss
from src.utilities.data_utils import load_dataset
from src.utilities.training_utils import train
from src.modules.loss_function import Loss
from src.modules.sampler import BimatrixSampler

def main():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument('--training_set', type=str, default=None, help="Training set (overrides config.toml)")
    parser.add_argument('--n_games', type=int, default=2**25, help="Number of games to train on")
    parser.add_argument('--config', type=str, default="2x2_example")    
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training")
    parser.add_argument('--optimizer', type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer (Adam or SGD)")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--gamma', type=float, default=1, help="Decay rate for LR scheduler")
    parser.add_argument('--init_model', type=str, default=None, help="Pre-trained model")    
    parser.add_argument('--log_models',  action='store_true', help="Log models")    
    parser.add_argument('--name', type=str, default=None, help="Model name")
    parser.add_argument('--seed', type=int, default=1, help="Seed")
 
    # Process configs
    args = parser.parse_args()
    config = read_config('config.toml', args.config)
    config, training_set = process_config(args, config)

    # Setting seed
    torch.manual_seed(args.seed)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Initialize row player network
    model1 = initialize_model(config['model1'], device)
    optimizer1 = initialize_optimizer(model1, args.optimizer, args.lr)
    scheduler1 = initialize_scheduler(optimizer1, args.gamma)
   
    # Initialize column player network
    model2 = initialize_model(config['model2'], device)
    optimizer2 = initialize_optimizer(model2, args.optimizer, args.lr)
    scheduler2 = initialize_scheduler(optimizer2, args.gamma)

    # Loss function
    loss_function = Loss(**config['loss'])
    
    # Bimatrix sampler
    rand_bimatrix = BimatrixSampler(**config['bimatrix'], set_games = training_set, device=device)

    # Generate metadata
    metadata, timestamp = generate_metadata(config, args)
    print("\nModel Metadata:")
    print_metadata(**metadata)

    # Train the model
    print("\nStarting training...\n")
    model1, model2, avg_regrets = train(model1, optimizer1, scheduler1, model2, optimizer2, scheduler2,
                                        args.n_games, args.batch_size, loss_function, rand_bimatrix, 
                                        timestamp * args.log_models)

    # Save trained models and simulation metadata
    model_path = os.path.join("models", args.name if args.name else timestamp)
    save_model(model1, model_path, file_name="model1", metadata=metadata['model1'], verbose=True)
    save_model(model2, model_path, file_name="model2", metadata=metadata['model2'], verbose=True)
    save_metadata(f'{model_path}/metadata.json', metadata)

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

