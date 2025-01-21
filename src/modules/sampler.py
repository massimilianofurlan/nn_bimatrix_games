import torch

class BimatrixSampler:
    def __init__(self, n_actions: int, payoffs_space: str = "sphere", game_class: str = "general_sum", 
                       set_games = None, device: str = 'cpu', dtype=torch.float32):
        self.n_actions = n_actions  
        self.game_class = game_class
        self.payoffs_space = payoffs_space
        self.device = device
        self.set_games = set_games
        self.dtype = dtype
        
        # number of payoffs for each agent
        self.n_payoffs = self.n_actions**2
        # householder rotation matrix
        self.H = self.householder_matrix(self.n_payoffs)
        # normal vector (for halfsphere)
        self.v_norm, self.u_norm = self.normal_vectors(self.n_payoffs)
        
        # define matrix sampling function based on payoffs_space
        if self.payoffs_space == 'sphere':
            self.sampler_matrix = self.rand_sphere
        #elif self.payoffs_space == 'halfsphere':
        #    self.sampler_matrix = self.rand_halfsphere
        elif self.payoffs_space == 'sphere_orthogonal':
            self.sampler_matrix = self.rand_sphere_orthogonal
        elif self.payoffs_space == 'hemisphere_orthogonal':
            self.sampler_matrix = self.rand_hemisphere_orthogonal
        elif self.payoffs_space == 'halfsphere_orthogonal':
            self.sampler_matrix = self.rand_halfsphere_orthogonal

        # define bimatrix sampling function based on game class
        if self.game_class == "general_sum":
            self.sampler_bimatrix = self.rand_generalsum_bimatrix
        elif self.game_class == "zero_sum":
            self.sampler_bimatrix = self.rand_zerosum_bimatrix
        elif self.game_class == "symmetric":
            self.sampler_bimatrix = self.rand_symmetric_bimatrix
        #elif self.game_class == "potential":
        #    self.sampler_bimatrix = self.sample_potential_bimatrix
        
        # exception when set_game is provided
        if self.set_games is not None:
            self.set_games = self.set_games.to(self.device)
            self.sampler_bimatrix = self.rand_from_set_bimatrix
    
    def householder_matrix(self, n):
        # householder rotation matrix mapping (1,...,1) to (0,...,0,sqrt(n))
        u = torch.ones(n, device=self.device, dtype=self.dtype, requires_grad=False)
        u_target = torch.zeros(n, device=self.device, dtype=self.dtype)
        u_target[-1] = torch.linalg.norm(u)
        v = u - u_target
        H = torch.eye(n, device=self.device, dtype=self.dtype) - 2 * torch.outer(v, v) / torch.dot(v, v)
        return H

    def normal_vectors(self, n):
        # normal vectors orthogonal to (1,1,...,1) and to each other
        # v=(-1,-1,...,1,1)
        v = torch.ones(n, device=self.device, dtype=self.dtype)
        v[:n//2] = -1.0
        v /= torch.linalg.vector_norm(v)
        # u=(-1,+1,-1,+1,...)
        u = torch.ones(n, device=self.device, dtype=self.dtype)
        u[::2] = -1.0
        u /= torch.linalg.vector_norm(u)
        return v, u        

    def rand_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in R^{n}
        x = torch.randn((k, n), device=self.device, dtype=self.dtype, requires_grad=False)
        norm = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-8)
        x.div_(norm).mul_(r)
        return x
    
    #def rand_halfsphere(self, k, n, r):
    #    # sample uniformly k points from a slice of r-radius sphere in R^{n} with measure 1/2
    #    x = self.rand_sphere(k, n, r)
    #    # x=(x_i) with x_1 > 0; 1/2 of the sphere
    #    # restrict to x=(x_i) with sign(x_{n-1}) = sign(x_n)
    #    x[:,n-2:n] = x[:,n-2:n].abs_()
    #    flip_sign = torch.where(torch.rand(k, device = self.device) > 1/2, -1.0, 1.0)
    #    x[:,n-2:n] = x[:,n-2:n].mul_(flip_sign.unsqueeze(1))     
    #    return x
    
    def rand_sphere_orthogonal(self, k, n, r):
        # sample uniformly k points from r-radius sphere in the subspace orthogonal to (1,...,1) in R^{n_actions^2}
        # note: (x)_i has variance r^2 / (n^2 - 1)
        # sample y uniformly from r-radius sphere in R^{n-1}
        y = self.rand_sphere(k, n - 1, r)
        # define z = (y.T,0)
        z = torch.zeros(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        z[:, :n - 1] = y
        # apply householder rotation mapping (1,...,1) to (0,...,sqrt(n))
        x = torch.matmul(z, self.H.T)
        # x is uniform in {x \in R^{n} | sum(x)=0, ||x||=r}
        return x
    
    def rand_hemisphere_orthogonal(self, k, n, r):
        # sample uniformly k points from r-radius hemisphere in the subspace orthogonal to (1,...,1) in R^{n_actions^2} 
        # hemisphere on the positive halfspace x^Tv>0
        # sample uniformly from r-radius sphere in subspace orthogonal to (1,...,1)
        x = self.rand_sphere_orthogonal(k, n, r)
        # positive halfspace x^Tv>0 with v = self.v_norm
        inners = torch.matmul(x, self.v_norm)
        pos_half_mask = inners > 0
        # reflect points on negative halfsphere across x^Tv=0 plane x <- x - 2 (x^Tv)v 
        x[~pos_half_mask] = x[~pos_half_mask] - 2 * torch.outer(inners[~pos_half_mask], self.v_norm)
        # x is uniform in {x \in R^{n} | sum(x)=0, ||x||=r, x^Tv>0}
        return x

    def rand_halfsphere_orthogonal(self, k, n, r):
        # sample uniformly k points from r-radius halfsphere in the subspace orthogonal to (1,...,1) in R^{n_actions^2} 
        # halfsphere on the positive and negative orthant sign(x^Tv) = sign(x^Tu)
        # sample uniformly from r-radius sphere in subspace orthogonal to (1,...,1)
        x = self.rand_sphere_orthogonal(k, n, r)
        # positive halfspace x^Tv>0 with v = self.v_norm
        inners_v = torch.matmul(x, self.v_norm)
        pos_half_v_mask = inners_v > 0
        # positive halfspace x^Tu>0 with u = self.u_norm
        inners_u = torch.matmul(x, self.u_norm)
        pos_half_u_mask = inners_u > 0
        # +- orthant x^Tv>0 and x^Tu<0
        posneg_orth_mask = pos_half_v_mask & ~pos_half_u_mask
        # -+ orthant x^Tv<0 and x^Tu>0
        negpos_orth_mask = ~pos_half_v_mask & pos_half_u_mask
        # reflect points on +- orthant across x^Tu=0 plane into the ++ orthant x <- x - 2 (x^Tu)u
        x[posneg_orth_mask] = x[posneg_orth_mask] - 2 * torch.outer(inners_u[posneg_orth_mask], self.u_norm)
        # reflect points on -+ orthant across x^Tu=0 plane into the -- orthant x <- x - 2 (x^Tu)u
        x[negpos_orth_mask] = x[negpos_orth_mask] - 2 * torch.outer(inners_u[negpos_orth_mask], self.u_norm)
        # x is uniform in {x \in R^{n} | sum(x)=0, ||x||=r, sgn(x^Tv)=sgn(x^Tu)}
        return x

    def rand_generalsum_bimatrix(self, batch_size):
        # sample general-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        B_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions)
        B = B_vec.view(batch_size, self.n_actions, self.n_actions)
        return A, B

    def rand_zerosum_bimatrix(self, batch_size):
        # sample zero-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions)
        return A, -A

    def rand_symmetric_bimatrix(self, batch_size):
        # sample symmetric bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions)
        return A, A.transpose(1,2)

    def rand_from_set_bimatrix(self, batch_size):
        # sample bimatrix game from set_games
        idx = torch.randint(self.set_games.size(0), (batch_size,))
        A, B = self.set_games[idx][:,0,:,:], self.set_games[idx][:,1,:,:]
        return A, B

    def __call__(self, batch_size: int) -> torch.Tensor:
        A, B = self.sampler_bimatrix(batch_size)
        G = torch.stack((A, B), dim=1)
        return G
