from src.utilities.bimatrix_utils import *
from src.utilities.data_utils import read_games_from_file
from src.utilities.io_utils import print_bimatrix_game
from src.modules.sampler import BimatrixSampler

# Initialize random bimatrix game sampler
rand_bimatrix = BimatrixSampler(payoffs_space='sphere', n_actions=4)
n_games = 2048
games = rand_bimatrix(n_games).numpy()

def test_expected_payoffs(G):
    A, B = G
    n_actions = G.shape[2]
    
    # pure strategy expected payoff
    x, y = np.random.rand(n_actions), np.random.rand(n_actions)
    x, y = x == x.max(), y == y.max()
    assert (A[x, y], B[x, y]) == get_expected_payoffs(G, x, y), \
        f"Test failed: wrong expected payoff for pure strategy profile {(x,y)}\nGame:\n{print_bimatrix_game(A, B)}"
    
    # mixed strategy expected payoff
    x, y = np.ones(n_actions, dtype=np.float32)/n_actions, np.ones(n_actions, dtype=np.float32)/n_actions
    ep1, ep2 = get_expected_payoffs(G, x, y)
    assert np.isclose(ep1, np.mean(A), atol=1e-7) and np.isclose(ep2, np.mean(B), atol=1e-7), \
        f"Test failed: wrong expected payoff for mixed strategy profile {(x,y)}\nGame:\n{print_bimatrix_game(A, B)}"

def test_dominated_and_rationalizable(G):
    A, B = G
    
    # Test dominated strategies
    dominated_mask_A, _ = get_dominated_mask(A)
    dominated_mask_B, _ = get_dominated_mask(B.T)
    
    # not all strategies dominated
    assert not np.all(dominated_mask_A) and not np.all(dominated_mask_B), \
        f"Test failed: all strategies dominated.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Test that Nash equilibria contain no dominated strategies
    set_nash, _ = get_nash_equilibria(G, rational=False)
    for x, y in set_nash:
        assert not np.any(dominated_mask_A[np.where(x)]) and not np.any(dominated_mask_B[np.where(y)]), \
            f"Test failed: Nash strategy is dominated.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Test rationalizable strategies
    rationalizable_mask = get_rationalizable_mask(G)
    
    # not all strategies non-rationalizable
    assert np.any(rationalizable_mask[0]) and np.any(rationalizable_mask[1]), \
        f"Test failed: All strategies non-rationalizable.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Test that Nash equilibria are rationalizable
    for x, y in set_nash:
        assert np.all(rationalizable_mask[0][np.where(x)]) and np.all(rationalizable_mask[1][np.where(y)]), \
            f"Test failed: Nash strategy is not rationalizable.\nGame:\n{print_bimatrix_game(A, B)}"

def test_nash_equilibria(G):
    A, B = G
    set_nash, set_payoffs = get_nash_equilibria(G, rational=False)
    
    # Check if the number of Nash equilibria is odd 
    n_nash = len(set_nash)
    assert n_nash % 2 == 1, f"Test failed: Expected an odd number of Nash equilibria, found {n_nash}\nGame:\n{print_bimatrix_game(A, B)}"
    
    for nash in set_nash:
        x, y = nash
        
        # regret must be zero
        regret1 = np.max(np.dot(A, y)) - np.dot(np.dot(x, A), y)
        regret2 = np.max(np.dot(x.T, B)) - np.dot(np.dot(x, B), y)
        assert np.isclose(regret1, 0, atol=1e-7) and np.isclose(regret2, 0, atol=1e-7), \
            f"Test failed: regrets = ({regret1}, {regret2})\nGame:\n{print_bimatrix_game(A, B)}"
        
        # strategies are valid probabilities
        assert np.isclose(np.sum(x), 1, atol=1e-7) and np.all(x >= 0) and np.isclose(np.sum(y), 1, atol=1e-7) and np.all(y >= 0), \
            f"Test failed: Nash strategy is not a valid probability distribution.\nGame:\n{print_bimatrix_game(A, B)}"

def test_maxmin_payoff(G):
    A, B = G
    
    # Compute maxmin payoffs for the original game
    maxmin_payoff = get_maxmin_payoff(G)
    
    # Compute Nash equilibria and payoffs
    set_nash, set_payoffs = get_nash_equilibria(G, rational=False)
    
    # Check that Nash payoffs are >= maxmin payoffs in general games
    for payoff in set_payoffs:
        assert np.all(payoff >= maxmin_payoff - 1e-7), \
            f"Test failed: Nash payoff {payoff} is less than maxmin payoff {maxmin_payoff}.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Convert the game to a zero-sum game by setting B = -A
    G_zero_sum = np.stack([A, -A])
    
    # Compute Nash equilibria and payoffs for the zero-sum game
    set_nash_zero_sum, set_payoffs_zero_sum = get_nash_equilibria(G_zero_sum, rational=False)
    
    # Compute maxmin payoffs for the zero-sum game
    maxmin_payoff_zero_sum = get_maxmin_payoff(G_zero_sum)
    
    # Ensure Nash payoffs equal the maxmin payoffs in the zero-sum game
    for payoff in set_payoffs_zero_sum:
        assert np.all(np.isclose(payoff, maxmin_payoff_zero_sum, atol=1e-7)), \
            f"Test failed: Nash payoff {payoff} does not equal maxmin payoff {maxmin_payoff_zero_sum} in zero-sum game.\nGame:\n{print_bimatrix_game(A, -A)}"

def test_pareto_optimal_nash_mask(G):
    A, B = G
    
    # Compute Nash equilibria and their corresponding payoffs
    set_nash, set_payoffs = get_nash_equilibria(G, rational=False)
    
    # Get Pareto optimal, utilitarian, and payoff-dominant masks
    pareto_optimal_nash, utilitarian_nash, payoff_dominance = get_pareto_optimal_nash_mask(G, set_payoffs)
    
    # Check that if utilitarian then it must also be Pareto optimal
    assert np.all(pareto_optimal_nash[utilitarian_nash]), \
        f"Test failed: Some utilitarian equilibria are not Pareto optimal.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Check that if payoff-dominant then it must also be utilitarian and Pareto optimal
    if np.any(payoff_dominance == 1):
        assert np.all(utilitarian_nash[payoff_dominance == 1]), \
            f"Test failed: Payoff-dominant equilibria are not utilitarian.\nGame:\n{print_bimatrix_game(A, B)}"
        assert np.all(pareto_optimal_nash[payoff_dominance == 1]), \
            f"Test failed: Payoff-dominant equilibria are not Pareto optimal.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Check that at least one Pareto optimal Nash equilibrium exists
    assert np.any(pareto_optimal_nash), \
        f"Test failed: No Pareto optimal Nash equilibrium found.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Check that at least one utilitarian Nash equilibrium exists
    assert np.any(utilitarian_nash), \
        f"Test failed: No utilitarian Nash equilibrium found.\nGame:\n{print_bimatrix_game(A, B)}"
    
    # Additional check: if there is only one Nash equilibrium, it should be all three
    if len(set_nash) == 1:
        assert pareto_optimal_nash[0] and utilitarian_nash[0] and payoff_dominance[0] == 1, \
            f"Test failed: Single Nash equilibrium is not classified correctly.\nGame:\n{print_bimatrix_game(A, B)}"


def run_test(test_function, games, test_name):
    for idx, G in enumerate(games):
        print(f"\rTesting {test_name} {idx+1}/{len(games)}", end='', flush=True)
        test_function(G)
    print(" [ok]")

# Example usage for each test function:
run_test(test_expected_payoffs, games, "expected payoffs")
run_test(test_dominated_and_rationalizable, games, "dominated and rationalizable strategies")
run_test(test_nash_equilibria, games, "Nash equilibria")
run_test(test_maxmin_payoff, games, "maxmin payoff")
run_test(test_pareto_optimal_nash_mask, games, "Pareto optimal Nash mask")
