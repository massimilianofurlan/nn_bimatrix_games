import torch

# NOTE: this sampler generates bimatrix games of the form (A,B^T)

class BimatrixSampler:
    def __init__(self, n_actions: int, payoffs_space: str = "sphere_preferences", game_class: str = "general_sum", 
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
        self.Hpr, self.Hbr = self.generate_rotations(self.n_payoffs)
        
        # define matrix sampling function based on payoffs_space
        if self.payoffs_space == 'sphere':
            self.sampler_matrix = self.rand_sphere
        elif self.payoffs_space == 'sphere_preferences':
            self.sampler_matrix = self.rand_preferences_sphere
        elif self.payoffs_space == 'sphere_strategic':
            self.sampler_matrix = self.rand_strategic_sphere
        
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
    
    def generate_rotations(self, n):
        # generate householder rotation for space of preferences (1 rotation)
        u = torch.ones(n, device=self.device, dtype=self.dtype)
        u_target = torch.zeros(u.shape[0], device=self.device, dtype=self.dtype)
        u_target[-1] = torch.linalg.norm(u)
        v = u - u_target
        Hpr = torch.eye(u.shape[0], device=self.device, dtype=self.dtype) - 2 * torch.outer(v, v) / torch.dot(v, v)
        # generate householder rotation for space of best-reply (n rotations)
        Hbr = torch.eye(n, device=self.device, dtype=self.dtype)
        for k in range(self.n_actions):
            v = torch.zeros(n, device=self.device, dtype=self.dtype)
            v[k * self.n_actions:(k + 1) * self.n_actions] = 1.0
            v_target = torch.zeros_like(v)
            v_target[self.n_actions*k] = torch.linalg.norm(v)  # Move to canonical basis
            w = v - v_target
            H = torch.eye(n, device=self.device, dtype=self.dtype) - 2 * torch.outer(w, w) / torch.dot(w, w)
            Hbr = H @ Hbr  # Apply the reflection
        return Hpr.requires_grad_(False), Hbr.requires_grad_(False)
    
    def rand_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in R^{n}
        x = torch.randn((k, n), device=self.device, dtype=self.dtype, requires_grad=False)
        norm = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-8)
        x.div_(norm).mul_(r)
        return x
    
    def rand_preferences_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in the subspace orthogonal to (1,...,1) in R^{n_actions^2}
        # note: (x)_i has variance 1
        # sample y uniformly from r-radius sphere in R^{n-1}
        y = self.rand_sphere(k, n - 1, r)
        # define z = (y.T,0)
        z = torch.zeros(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        z[:, :n - 1] = y
        # apply householder rotation mapping (1,...,1) to (0,...,sqrt(n))
        x = torch.matmul(z, self.Hpr.T)
        # x is uniform in {x \in R^{n} | sum(x)=0, ||x||=r}
        return x
    
    def rand_strategic_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in the subspace 
        # orthogonal to {(1,0,..),(0,1,0,...),...} (where 1 and 0 are 1xn)
        # note: (x)_i has variance 1
        # sample y uniformly from r-radius sphere in R^{n-n_actions}
        y = self.rand_sphere(k, n - self.n_actions, r)
        # define z.view(n,n)[n-1,n] = y.view(n,n)
        z = torch.zeros(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        z[:, torch.arange(n) % self.n_actions != 0] = y
        # apply householder rotation
        x = torch.matmul(z, self.Hbr.T)
        # x is uniform in {x \in R^{n} | 1^T x.view(n,n)=0, ||x||=r}
        return x
    
    def rand_generalsum_bimatrix(self, batch_size):
        # sample general-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        B_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # transpose because torch is col-major
        B = B_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # transpose because torch is col-major
        return A, B
    
    def rand_zerosum_bimatrix(self, batch_size):
        # sample zero-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # transpose because torch is col-major
        return A, -A
    
    def rand_symmetric_bimatrix(self, batch_size):
        # sample symmetric bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # transpose because torch is col-major
        return A, A.transpose(1,2)
    
    def rand_from_set_bimatrix(self, batch_size):
        # sample bimatrix game from set_games with convention (A,B^T)
        idx = torch.randint(self.set_games.size(0), (batch_size,))
        A, B = self.set_games[idx][:,0,:,:], self.set_games[idx][:,1,:,:]
        return A, B
    
    def __call__(self, batch_size: int) -> torch.Tensor:
        A, B = self.sampler_bimatrix(batch_size)
        G = torch.stack((A, B), dim=1)
        return G
