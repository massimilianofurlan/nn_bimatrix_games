import torch
from src.modules.sampler import BimatrixSampler

def test_bimatrix_sampler():
    n_actions = 3
    batch_size = 2**25

    def print_result(test_name, passed, expected, actual):
        """Prints test results with expected and actual values."""
        if passed:
            print(f"{test_name} passed [ok]")
        else:
            print(f"{test_name} FAILED [!!] Expected: {expected}, Got: {actual}")

    # Test 1: Sphere Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere", device='cpu')
    G = sampler(batch_size)
    A, B = G[:, 0], G[:, 1]

    norm_A = torch.norm(A.view(batch_size, -1), dim=1)
    norm_B = torch.norm(B.view(batch_size, -1), dim=1)

    expected_norm = n_actions
    max_dev_A = torch.max(abs(norm_A - expected_norm)).item()
    max_dev_B = torch.max(abs(norm_B - expected_norm)).item()
    passed = max_dev_A < 1e-5 and max_dev_B < 1e-5

    print_result("Sphere Sampling Norm Check", passed, expected_norm, f"A: {norm_A.mean().item()}, B: {norm_B.mean().item()}")

    # Test 2: Sphere Preferences Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere_preferences", device='cpu')
    G = sampler(batch_size)

    sum_G = G.sum(dim=(2, 3))
    max_sum = sum_G.abs().max().item()
    passed = max_sum < 1e-5
    print_result("Sphere Preferences Sum Constraint", passed, "≈0", max_sum)

    std_dev = G.std(dim=(0, 1))
    max_dev = torch.max(abs(std_dev - 1)).item()
    passed = max_dev < 1e-3
    print_result("Sphere Preferences Standard Deviation", passed, "≈1", std_dev.mean().item())

    expected_max = (n_actions**2 - 1) ** 0.5
    max_abs_G = max(G.max().item(), abs(G.min().item()))
    deviation = expected_max - max_abs_G
    passed = abs(deviation) < 1e-3
    print_result("Sphere Preferences Max-Min Check", passed, expected_max, max_abs_G)

    # Test 3: Sphere Strategic Sampling
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere_strategic", device='cpu')
    G = sampler(batch_size)

    max_sum = max(G[:,0,].sum(dim=1).abs().max().item(), G[:,1,].sum(dim=2).abs().max().item())
    passed = max_sum < 1e-5
    print_result("Sphere Strategic Column Sum Constraint", passed, "≈0", max_sum)

    std_dev = G.std(dim=(0, 1))
    max_dev = torch.max(abs(std_dev - 1)).item()
    passed = max_dev < 1e-3
    print_result("Sphere Strategic Standard Deviation", passed, "≈1", std_dev.mean().item())

    expected_max = (n_actions**2 - n_actions) ** 0.5
    max_abs_G = max(G.max().item(), abs(G.min().item()))
    deviation = expected_max - max_abs_G
    passed = abs(deviation) < 1e-3
    print_result("Sphere Strategic Max-Min Check", passed, expected_max, max_abs_G)

    norm_G = G.norm(dim=(2, 3))
    max_dev = torch.max(abs(norm_G - n_actions)).item()
    passed = max_dev < 1e-5
    print_result("Sphere Strategic Norm Check", passed, n_actions, norm_G.mean().item())

    # Test 4: Sphere Preferences Subspaces
    n_actions = 2
    v_norm_A = torch.tensor([-1.0, 1.0, 1.0, -1.0], device='cpu').detach()
    v_norm_B = torch.tensor([1.0, -1.0, -1.0, 1.0], device='cpu').detach()
    sampler = BimatrixSampler(n_actions=n_actions, payoffs_space="sphere_preferences", device='cpu',
                          normal_vectors = [v_norm_A, v_norm_B])

    G = sampler(batch_size)
    sum_G = G.sum(dim=(2, 3))
    max_sum = sum_G.abs().max().item()
    passed = max_sum < 1e-5
    print_result("Sphere Preferences Subspaces Sum Constraint", passed, "≈0", max_sum)

    std_dev = G.std(dim=(0, 1))
    max_dev = torch.max(abs(std_dev - 1)).item()
    passed = max_dev < 1e-3
    print_result("Sphere Preferences Subspaces Standard Deviation", passed, "≈1", std_dev.mean().item())

    expected_max = (n_actions**2 - 1) ** 0.5
    max_abs_G = max(G.max().item(), abs(G.min().item()))
    deviation = expected_max - max_abs_G
    passed = abs(deviation) < 1e-3
    print_result("Sphere Preferences Subspaces Max-Min Check", passed, expected_max, max_abs_G)

    A_vec = G[:,0,:,:].permute(0,2,1).reshape(-1, n_actions**2)
    B_vec = G[:,1,:,:].reshape(-1, n_actions**2)
    inners_A = torch.matmul(A_vec, v_norm_A)
    inners_B = torch.matmul(B_vec, v_norm_B)
    min_inners = min(inners_A.amax(), inners_B.amax())
    passed = min_inners > -1e-5
    print_result("Sphere Preferences Subspaces Inners Check", passed, "≈0", min_inners)


# Run the tests
test_bimatrix_sampler()