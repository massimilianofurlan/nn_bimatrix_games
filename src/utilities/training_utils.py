import os
import time
import torch
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from src.modules.loss_function import Loss
from src.modules.sampler import BimatrixSampler
from src.utilities.model_utils import save_model

def transpose_game(G):
    # input G=(A,B) outputs G'=(B',A')
    A, B = G[:,0,:,:], G[:,1,:,:]
    G_transpose = torch.stack((B.transpose(1, 2), A.transpose(1, 2)), dim=1)
    return G_transpose

def optimize_model(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
   

def grad_norm(model: torch.nn.Module) -> float:
    """ compute the L2 norm of the accumulated gradients. """
    total_norm = sum(x.grad.data.norm(2) ** 2 for x in model.parameters() if x.grad is not None)**(1/2)
    return total_norm#.item()


def print_epoch_stats(step: int, log_interval: int, batch_size: int, avg_regrets: float, model: torch.nn.Module, 
                      optimizer: Optimizer, start_time: float) -> None:
    """ print statistics for the current epoch. """
    avg_regret = avg_regrets[step+1-log_interval:step+1,:].mean()
    avg_grad_norm = grad_norm(model)
    log_step = (step+1) // log_interval
    games_processed = (step+1) * batch_size
    current_lr = optimizer.param_groups[0]['lr']
    duration = time.time() - start_time
    print(f'Step {log_step}/128, Avg. Regret: {avg_regret:.4f}, '
          f'Avg. Grad. Norm: {avg_grad_norm:.3f}, Lr: {current_lr:.1e}, '
          f'Time: {duration:.2f}, Games: {games_processed}')    


def train(model1: torch.nn.Module, optimizer1: Optimizer, scheduler1: _LRScheduler, 
          model2: torch.nn.Module, optimizer2: Optimizer, scheduler2: _LRScheduler, 
          n_games: int, batch_size: int, loss_function: Loss, rand_bimatrix: BimatrixSampler, timestamp: str) -> list[float]:
    """
    Args:
        modeli (torch.nn.Module): Player i's model for to train.
        optimizeri (Optimizer): Player i's  optimizer for training.
        scheduleri (_LRScheduler): Player i's learning rate scheduler.
        n_games (int): Number of games to simulate.
        batch_size (int): Number of games in each batch.
        timestamp (str): Timestamp for saving the model (= to '' if log_models = false)

    Returns:
        list[float]: List of average regrets per step.
    """
    
    n_optimization_steps = n_games // batch_size
    log_interval = n_optimization_steps // 128    

    avg_regrets = np.empty((n_optimization_steps,2), dtype=np.float32)

    log_models = bool(timestamp)
    if log_models:
        # Generate folder
        models_log_path = os.path.join("models", timestamp, "models_log")
        # Define log steps (approximately uniformly spaced on log10 scale)
        exp = np.ceil(np.log10(n_optimization_steps)).astype(int)
        model_log_steps = np.logspace(0, exp, num=20*exp+1, dtype=int)

    start_time = time.time()
    for step in range(0, n_optimization_steps):
        # Generate batch_size bimatrix games G=(A,B)
        G = rand_bimatrix(batch_size)  # shape: batch_size x 2 x n_actions x n_actions
        # Derive column player perspective G'=(B',A')
        G_transpose = transpose_game(G)

        # Forward pass
        p = model1(G)
        q = model2(G_transpose)

        # Compute losses
        regret1, loss1 = loss_function(G, p, q.detach())
        regret2, loss2 = loss_function(G_transpose, q, p.detach())

        optimize_model(optimizer1, loss1)
        optimize_model(optimizer2, loss2)
        scheduler1.step()
        scheduler2.step()

        avg_regrets[step,0] = regret1.mean().item()
        avg_regrets[step,1] = regret2.mean().item()
        
        if log_models and step in model_log_steps: 
            save_model(model1, models_log_path, file_name=f'model1_{step:.0f}.pth')
            save_model(model2, models_log_path, file_name=f'model2_{step:.0f}.pth')

        if (step+1) % log_interval == 0:
            print_epoch_stats(step, log_interval, batch_size, avg_regrets, model1, optimizer1, start_time)
            start_time = time.time()

    avg_regrets = np.array(avg_regrets, dtype=np.float16)
    return model1, model2, avg_regrets

