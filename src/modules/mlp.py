import torch
import torch.nn as nn
import torch.nn.init as init

'''
class Normalize(nn.Module):
    def __init__(self, n_actions: int):
        super(Normalize, self).__init__()
        self.n_actions = n_actions
    
    def forward(self, G: torch.Tensor) -> torch.Tensor:
        # Demean payoffs: x <- x - mean(x)
        G = G - torch.mean(G, dim=(2, 3), keepdim=True)
        # Normalize payoffs into unit sphere x <- x / norm(x)
        norm = torch.linalg.matrix_norm(G, dim=(2, 3), keepdim=True)
        norm = norm.where(norm > 0, torch.tensor(1.0, device=norm.device))
        G = G / norm # G.div_(norm) inplace, fast, unsafe
        # Unit variance (sphere of radius n_actions) x <- x * n_actions
        G = G * self.n_actions # G.mul_(n_actions) inplace, fast, unsage
        return G
'''

class MLP_Bimatrix(nn.Module):
    """
    Multi-layer perceptron model to play bimatrix games.

    Args:
        n_actions (int): Number of actions in the game.
        n_layers (int): Number of hidden layers in the MLP.
        hidden_dim (int): Dimension of the hidden layers.
    """
    def __init__(self, n_actions: int, n_layers: int, hidden_dim: int):
        super().__init__()
        self.n_actions = n_actions
        
        modules = []
        # input layer
        modules.append(nn.Flatten())
        # hidden layers
        modules.append(nn.Linear(2*n_actions**2, hidden_dim))
        modules.append(nn.ReLU(inplace=True))
        for _ in range(n_layers-1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.ReLU(inplace=True))
        # output layer
        modules.append(nn.Linear(hidden_dim, n_actions))
        modules.append(nn.Softmax(dim=1))
        
        # combine layers
        self.network = nn.Sequential(*modules)
        # initialization
        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            init.zeros_(module.bias)
    
    def forward(self, G: torch.Tensor) -> torch.Tensor:
        return self.network(G)
