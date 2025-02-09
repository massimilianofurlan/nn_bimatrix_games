import torch

class BimatrixSampler:
    def __init__(self, n_actions: int, payoffs_space: str = "sphere_preferences", 
                       game_class: str = "general_sum", set_games: torch.Tensor | None = None,
                       normal_vectors = [[], []], device: str = 'cpu', dtype=torch.float32):
        """
        A sampler for generating bimatrix games G = (A, B).

        Args:
            n_actions (int): number of actions per player (>1)
            payoffs_space (str): space of payoff (sphere, preferences, strategic)
            game_class (str): type of game (general sum, zero sum, symmetric)
            set_games (torch.Tensor, optional): optional predefined set of games 
            normal_vectors (tuple[torch.Tensor or None, torch.Tensor or None], optional): 
                normal vectors defining the hyperplanes that partition the space of games,
            device (str): device
            dtype (torch.dtype): data type (float32 or float64)
        """
        self.n_actions = n_actions  
        self.game_class = game_class
        self.payoffs_space = payoffs_space
        self.set_games = set_games
        self.normal_vectors = normal_vectors
        self.device = device
        self.dtype = dtype
        
        # number of payoffs for each agent
        self.n_payoffs = self.n_actions**2
        # householder rotation matrices
        self.Hpr, self.Hbr = self.generate_rotations(self.n_payoffs)
        # normal vectors
        self.v_norm_A, self.v_norm_B = self.generate_normal_vectors(self.normal_vectors)
        
        # define matrix sampling function based on payoffs_space (samples A)
        if self.payoffs_space == 'sphere':
            self.sampler_matrix = self.rand_sphere
        elif self.payoffs_space == 'sphere_preferences':
            self.sampler_matrix = self.rand_preferences_sphere
        elif self.payoffs_space == 'sphere_strategic':
            self.sampler_matrix = self.rand_strategic_sphere
        
        # define bimatrix sampling function based on game class (samples A,B)
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
            #v_target[self.n_actions*k] = torch.linalg.norm(v)
            v_target[n - self.n_actions + k] = torch.linalg.norm(v)
            w = v - v_target
            H = torch.eye(n, device=self.device, dtype=self.dtype) - 2 * torch.outer(w, w) / torch.dot(w, w)
            Hbr = H @ Hbr
        return Hpr.requires_grad_(False), Hbr.requires_grad_(False)
    
    def generate_normal_vectors(self, normal_vectors):
        v_norm_A, v_norm_B = normal_vectors
        v_norm_A = torch.as_tensor(v_norm_A, device=self.device, dtype=self.dtype)
        v_norm_B = torch.as_tensor(v_norm_B, device=self.device, dtype=self.dtype)
        v_norm_A /= 1 if v_norm_A.numel() == 0 else torch.linalg.norm(v_norm_A)
        v_norm_B /= 1 if v_norm_B.numel() == 0 else torch.linalg.norm(v_norm_B)
        return v_norm_A.requires_grad_(False), v_norm_B.requires_grad_(False)
    
    def reflect(self, x, v_norm):
        # reflect points in x^T v < 0 across the hyperplane orthogonal to v
        # v must have norm 1 (x <- x - 2 (x^Tv)v/(v'v) = x - 2 (x^Tv)v
        if v_norm.numel() == 0:
            return x
        # compute inners x^Tv
        inners = torch.matmul(x, v_norm)
        # mask positive halfspace x^Tv>0
        pos_half_mask = inners > 0
        # reflect points in x^T v < 0 across x^Tv=0 plane x <- x - 2 (x^Tv)v 
        x[~pos_half_mask] = x[~pos_half_mask] - 2 * torch.outer(inners[~pos_half_mask], v_norm)
        return x
    
    def rand_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in R^{n}
        x = torch.randn((k, n), device=self.device, dtype=self.dtype, requires_grad=False)
        norm = torch.norm(x, dim=1, keepdim=True).clamp_min(1e-8)
        x.div_(norm).mul_(r)
        # x is uniform in {x \in R^{n} | ||x||=r}
        return x
    
    def rand_preferences_sphere(self, k, n, r):
        # sample uniformly k points from r-radius sphere in the subspace orthogonal to (1,...,1) in R^{n_actions^2}
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
        # sample y uniformly from r-radius sphere in R^{n-n_actions}
        y = self.rand_sphere(k, n - self.n_actions, r)
        # define z.view(n,n)[n-1,n] = y.view(n,n)
        z = torch.zeros(k, n, device=self.device, dtype=self.dtype, requires_grad=False)
        #z[:, torch.arange(n) % self.n_actions != 0] = y
        z[:,:n - self.n_actions] = y
        # apply householder rotation
        x = torch.matmul(z, self.Hbr.T)
        # x is uniform in {x \in R^{n} | 1^T x.view(n,n)=0, ||x||=r}
        return x        
    
    def rand_generalsum_bimatrix(self, batch_size):
        # sample general-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        B_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A_vec = self.reflect(A_vec, self.v_norm_A)
        B_vec = self.reflect(B_vec, self.v_norm_B)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # torch vec() is row-major
        B = B_vec.view(batch_size, self.n_actions, self.n_actions)
        return A, B
    
    def rand_zerosum_bimatrix(self, batch_size):
        # sample zero-sum bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A_vec = self.reflect(A_vec, self.v_norm_A)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # torch vec() is row-major
        return A, -A
    
    def rand_symmetric_bimatrix(self, batch_size):
        # sample symmetric bimatrix game
        A_vec = self.sampler_matrix(batch_size, self.n_payoffs, self.n_actions)
        A_vec = self.reflect(A_vec, self.v_norm_A)
        A = A_vec.view(batch_size, self.n_actions, self.n_actions).transpose(1,2)    # torch vec() is row-major
        return A, A.transpose(1,2)
    
    def rand_from_set_bimatrix(self, batch_size):
        # sample bimatrix game from set_games {(A,B)_k}
        idx = torch.randint(self.set_games.size(0), (batch_size,))
        A, B = self.set_games[idx][:,0,:,:], self.set_games[idx][:,1,:,:]
        return A, B
    
    def __call__(self, batch_size: int) -> torch.Tensor:
        A, B = self.sampler_bimatrix(batch_size)
        G = torch.stack((A, B), dim=1)
        return G
