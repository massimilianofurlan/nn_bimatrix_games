import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

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

class Vectorize(nn.Module):
    def __init__(self, n_payoffs: int):
        super(Vectorize, self).__init__()
        self.n_payoffs = n_payoffs
    
    def forward(self, G: torch.Tensor) -> torch.Tensor:
        batch_size = G.size(0)
        G = G.reshape(batch_size, self.n_payoffs)
        return G

class MLP_Bimatrix(nn.Module):
    """
    Multi-layer perceptron model to play bimatrix games.

    Args:
        n_actions (int): Number of actions in the game.
        n_layers (int): Number of hidden layers in the MLP.
        hidden_dim (int): Dimension of the hidden layers.
        norm_layer (bool): Whether to apply normalization into (n**2-1)-sphere with radius n_actions centered at the origin.        
    """
    def __init__(self, n_actions: int, n_layers: int, hidden_dim: int, norm_layer: bool = False):
        super(MLP_Bimatrix, self).__init__()
        
        self.n_actions = n_actions
        
        # Preprocessing layer
        self.norm_layer = Normalize(n_actions) if norm_layer else nn.Identity()
        self.vec_layer = Vectorize(2*n_actions**2)
        
        # MLP layers
        self.fc_input = nn.Linear(2*n_actions**2, hidden_dim)
        self.fc_hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.fc_output = nn.Linear(hidden_dim, n_actions)
        
        # Activation function
        self.activation_fn = nn.ReLU()

        # Apply He Initialization to the layers
        # self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, nonlinearity='relu')
            init.zeros_(module.bias)
    
    def forward(self, G: torch.Tensor) -> torch.Tensor:
        x = G
        x = self.norm_layer(x)
        x = self.vec_layer(x)
        x = self.activation_fn(self.fc_input(x))
        for fc in self.fc_hidden_layers:
            x = self.activation_fn(fc(x))
        x = self.fc_output(x)
        x = F.softmax(x, dim=1)
        return x
