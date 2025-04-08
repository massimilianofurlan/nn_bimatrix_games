import torch

class Loss:
    def __init__(self, ex_ante: bool = True, p: int = 2):
        """
        Args:
            ex_ante (bool): whether to use ex-ante regret (default True)
                            if False, opponent's action is sampled from y
            p (int): exponent for the loss function (default 2)
        """
        self.ex_ante = ex_ante
        self.p = p
    
    @staticmethod
    def regret(G: torch.Tensor, x: torch.Tensor, y: torch.Tensor, grad=True) -> torch.Tensor:
        """
        calculate the regret of the row player with payoff matrix A from playing x against y.
        Args:
            G (torch.Tensor): bimatrix game
            x (torch.Tensor): mixed strategy for row player
            y (torch.Tensor): mixed strategy for col opponent
        Returns:
            torch.Tensor: returns row player regret of x against y r(G,x,y) = max_i [Ay]_i - x'Ay.
        """
        # extract the payoff matrix for the row player
        A = G[:, 0, :, :]
        # vector of expected payoffs from each pure strategy 
        Ay = torch.bmm(A, y.unsqueeze(2)).squeeze(2)  # shape: batch_size x n_actions
        # expected payoff from strategy x
        xAy = torch.bmm(x.unsqueeze(1), Ay.unsqueeze(2)).squeeze()  # shape: batch_size
        # expected payoff from best response to y
        Ay_max = Ay.max(dim=1).values  # shape: batch_size
        #Ay_max = ((Ay/1e-10).softmax(1) * Ay).sum(dim=1) 
        # regret max_i [Ay]_i - x'Ay 
        #grad = -regret.unsqueeze(1)*Ay/G.size(0) if compute_gradient else None
        return Ay_max - xAy  # shape: batch_size

    def __call__(self, G: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        compute regret and loss for the row player given bimatrix game G and strategy profile (x,y)
        Args:
            G (torch.Tensor): bimatrix game
            x (torch.Tensor): mixed strategy for the row player
            y (torch.Tensor): mixed strategy for the column player
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 
                - regret values for the row player, shape (batch_size,)
                - mean loss value over the batch for the row network
        """
        # compute regret values (opponent strategy detached from computational graph)
        regret_values = self.regret(G, x, y)
        # compute loss 
        if self.ex_ante:
            loss_values = regret_values.pow(self.p).mean()
        else:
            # sample opponent's strategy
            y_ex_post = torch.zeros_like(y).scatter_(1, torch.multinomial(y, 1), 1.0)
            regret_values_ex_post = self.regret(G, x, y_ex_post)
            loss_values = regret_values_ex_post.pow(self.p).mean()
        # return regret values and mean loss value over the batch
        return regret_values.detach(), loss_values.mean()
