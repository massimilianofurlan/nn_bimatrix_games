import torch
from src.modules.sampler import BimatrixSampler

def test_bimatrix_sampler():
    # Test parameters
    n_actions = 3
    batch_size = 2**14
    
    # Test 1: Sphere Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere", device='cpu')
    G = sampler(batch_size)
    A, B = G[:, 0], G[:, 1]
    
    norm_A = torch.norm(A.view(batch_size, -1), dim=1)
    norm_B = torch.norm(B.view(batch_size, -1), dim=1)
    
    assert torch.allclose(norm_A, torch.tensor([n_actions] * batch_size, dtype=torch.float32)) and \
           torch.allclose(norm_B, torch.tensor([n_actions] * batch_size, dtype=torch.float32)), \
           (f"Test failed: A or B is not sampled from a sphere with radius {n_actions}.\n"
            f"Max deviation A: {torch.max(norm_A - n_actions)}, B: {torch.max(norm_B - n_actions)}")
    
    print("Sphere sampling test passed [ok]")
      
    # Test 2: Sphere Orthogonal Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere_orthogonal", device='cpu')
    G = sampler(batch_size)
    A, B = G[:, 0], G[:, 1]
    
    max_sum = G.sum(dim=(2, 3)).max().item()
    assert max_sum <= 5e-6, \
           f"Test failed: A or B are not in the subspace orthogonal to (1,...,1). Max sum: {max_sum}"
    
    print("Sphere orthogonal sampling test passed [ok]")
        
    # Test 3: Zero-sum Game Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere", zero_sum=True, device='cpu')
    G = sampler(batch_size)
    A, B = G[:, 0], G[:, 1]
    
    assert torch.allclose(A, -B, atol=5e-6), \
           (f"Test failed: Zero-sum game does not satisfy A = -B.\n"
            f"Max deviation: {torch.max(A + B).item()}")
    
    print("Zero-sum game sampling test passed [ok]")

# Run the tests
test_bimatrix_sampler()