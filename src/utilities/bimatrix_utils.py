import numpy as np
import itertools
import pygambit
from lemke.bimatrix import bimatrix
from fractions import Fraction
from scipy.optimize import linprog

def get_expected_payoffs(G, x, y):
    # compute expected payoff for row (1) and col (2) player
    A, B = G
    # u1(p,q) = x'Ay 
    ep1 = np.dot(np.dot(x, A), y)
    # u1(p,q) = x'By
    ep2 = np.dot(np.dot(x, B), y)
    return ep1, ep2

def get_nash_equilibria(G, rational=True):
    # compute set of nash equilibria and set of payoffs
    A, B = G
    pygambit_game = pygambit.Game.from_arrays(A,B)
    pygambit_set_nash = pygambit.nash.enummixed_solve(pygambit_game, rational=rational).equilibria
    set_nash = []
    set_payoffs = []
    for pygambit_nash in pygambit_set_nash:
        x = [pygambit_nash['1'][z] for z in pygambit_game.players['1'].strategies]
        y = [pygambit_nash['2'][z] for z in pygambit_game.players['2'].strategies]
        set_nash.append([x,y])
        set_payoffs.append(get_expected_payoffs(G,x,y))
    return np.array(set_nash), np.array(set_payoffs)

def get_maxmin_payoff(G):
    # compute maxmin payoff of each player on nxn bimatrix game G
        
    # linear program: max_x v subject to (A^T)x >= ve, (x^T)e=1, x >= 0
    #                 where v is the a scalar, A is row player payoff matrix
    
    # linprog: min (x^T)c subject to A_ub x <= b_ub,  A_eq x = b_eq, lb <= x <= ub
    
    # lin prog input: c = (0,..,0,-1), A_ub = (A^T,e), b_ub = (0,..,0), 
    #                 A_eq = (e^T,0), b_eq = (e^T,0), lb = (0,0)
    #                 where e=(1,..,1)^T and maximizer is x = (x_, v)
    A, B = G
    n_actions, _ = A.shape
    
    # x is in the simplex
    A_eq = np.ones((1, n_actions+1))
    A_eq[:,-1] = 0.0
    b_eq = np.array([1])
    bounds = [(0.0, None) for _ in range(n_actions)]
    # no bounds for maxminval
    bounds.append((None,None))
    
    # -(A^T)x + e v <= 0   ->  (A^T)x >= e v
    c_A = np.zeros(n_actions+1)
    c_A[-1] = -1
    A_ub_A = np.hstack([-A.T, np.ones((n_actions,1))])
    b_ub_A = np.zeros(n_actions)
    
    # -(B)y + e v <= 0   ->  (B)y >= e v
    c_B = np.zeros(n_actions+1)
    c_B[-1] = -1
    A_ub_B = np.hstack([-B, np.ones((n_actions,1))])
    b_ub_B = np.zeros(n_actions)
    
    # lp for row player
    result1 = linprog(c_A, A_ub=A_ub_A, b_ub=b_ub_A, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    # lp for col player
    result2 = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    #maxmin_x = result1.x[0:n_actions]
    #maxmin_y = result2.x[0:n_actions]
    maxmin_payoff = np.array([result1.fun, result2.fun])
    return maxmin_payoff
        
def get_pure_nash_mask(set_nash):
    # returns a boolean mask for pure equilibria 
    return np.all(np.any(set_nash == 1, axis=2),axis=1)
 
def get_pareto_optimal_nash_mask(G, set_payoffs):
    # returns a boolean mask for pareto optimal equilirbia and utilitarian equilibria
    n = len(set_payoffs)
    pareto_optimal_nash = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and  np.all(set_payoffs[i] >= set_payoffs[j]) and not np.array_equal(set_payoffs[i], set_payoffs[j]):
                pareto_optimal_nash[j] = False
    
    # WIP 1: payoff dominant, 0 payoff dominated (or neither if all 0)
    payoff_dominance = np.ones(n, dtype=int) if n == 1 else np.zeros(n, dtype=int)  # if n = 1, vacuous
    for i in range(n):
        if all((set_payoffs[i] > set_payoffs[j]).all() for j in range(n) if i != j):
            # i is payof dominant
            payoff_dominance[i] = 1
            break
    
    # create boolean array with True at the index of max sum
    max_sums = np.sum(set_payoffs, axis=1)
    max_sum_indices = np.where(max_sums >= np.max(max_sums) - 1e-15)
    utilitarian_nash = np.zeros(n, dtype=bool)
    utilitarian_nash[max_sum_indices] = True
    
    return pareto_optimal_nash, utilitarian_nash, payoff_dominance


def get_dominated_mask(A, extent=False):
    # returns a boolean mask for dominated strategies (row player perspective)
    n_actions, n_actions_opponent = A.shape
    dominated_mask = np.zeros(n_actions, dtype=bool)
    
    # linprog solves max_z c'z s.t. Az<=b, Bz = d, lb <= x <= ub  
    
    # x is a mixed strategy, y >= 0 is a slack variable
    # objective is max_{x,y} y 
    # or equivalently  min_{x,y} a'x - by, with a' = (0,..,0) and b = -1
    # then c' = (a',b) = (0,..,0,-1)
    c = np.zeros(n_actions+1)
    c[-1] = -1
    
    # domination condition for si: \sum_{si'} x(si') u(si',sj) > ui(si,sj) for all sj
    # for strict domination: \sum_{si'} x(si') u(si',sj) >= ui(si,sj) + y for all sj
    # rewrites as \sum_{si'} x(si') [ui(si,sj) - u(si',sj)] + y <= 0 
    # domination condition for si in  matrix form is Az<=b, with
    # A[j,i'] = [ui(si,sj) - u(si',sj)]  ,  A[j,end] = 1  and  b = [0,..,0]
    A_ub = np.ones((n_actions_opponent, n_actions+1))
    b_ub = np.zeros(n_actions_opponent)
    
    # look in the simplex: \sum_{si'} x(si') = 1
    # since we are minimizing over the argument z = (x',y)'
    # constraint is written as (1,...,1,0) * z  = (1,...,1,0) * x' =  1
    A_eq = np.ones((1, n_actions+1))
    A_eq[:,-1] = 0.0
    b_eq = np.array([1])
    # probabilities and slack variable are non-negative
    bounds = [(0.0, None) for _ in range(n_actions+1)]
    
    min_payoff_diff = np.inf * np.ones(n_actions)
    for action in range(n_actions):
        for j, action_ in enumerate(range(n_actions)):
            A_ub[:,j] = A[action, :] - A[action_, :]
            if np.all(A_ub[:, j] < 0.0):
                min_payoff_diff[action] = np.min(-A_ub[:,j])
                dominated_mask[action] = True
                break
        else:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            if result.success and result.fun < 0.0:
                min_payoff_diff[action] = -result.fun
                dominated_mask[action] = True
    
    # min_payoff_diff captures domination extent in payoff terms
    return (dominated_mask, min_payoff_diff) if extent else dominated_mask


def get_rationalizable_mask(G, dominated_mask=None):
    # returns boolean mask for pure strategies surviving iterated elimination of strictly dominated strategies
    A, B = G
    n_actions1, n_actions2 = A.shape
    survived1_idxs = np.arange(n_actions1, dtype=np.int8)
    survived2_idxs = np.arange(n_actions2, dtype=np.int8)
    
    if dominated_mask is None:
        undominated1 = ~get_dominated_mask(A)
        undominated2 = ~get_dominated_mask(B.T)
    else:
        undominated1 = ~dominated_mask[0]
        undominated2 = ~dominated_mask[1]
    
    while True:
        # update indices of survived strategies
        survived1_idxs = survived1_idxs[undominated1]
        survived2_idxs = survived2_idxs[undominated2]
        # check if no strategies were eliminated in this iteration
        if undominated1.all() and undominated2.all():
            break
        # reduce the payoff matrices to only include undominated strategies
        A = A[undominated1,:][:, undominated2]
        B = B[:,undominated2][undominated1, :]
        # eliminate dominated strategies
        undominated1 = ~get_dominated_mask(A)
        undominated2 = ~get_dominated_mask(B.T)
    
    # survived strategies boolean arrays
    rationalizable_set1 = np.zeros(n_actions1, dtype=bool)
    rationalizable_set2 = np.zeros(n_actions2, dtype=bool)
    rationalizable_set1[survived1_idxs] = True
    rationalizable_set2[survived2_idxs] = True
    
    return np.array([rationalizable_set1, rationalizable_set2])


def get_indeces(G, set_nash):
    # returns stability index of each nash in the game 
    A, B = G
    n_actions1, n_actions2 = A.shape
    gambit_game = bimatrix(A, B)
    indeces = []
    for nash in set_nash:
        index = gambit_game.eqindex(nash.flatten(),n_actions1,n_actions2)
        indeces.append(index)

    return np.array(indeces)


def get_harsanyi_selten_mask(G, set_nash, n_trace = 1000):
    # returns mask of harsanyi selten linear tracing selected equilibrium
    # note: incorrectly selects a pure when symmetry-invariance requires the selected
    #       equilibrium to be a mixed equilibrium (measure zero).
    #       outcome is noisy (traces are close) when close to those games
    A, B = G
    gambit_game = bimatrix(A, B)
    n_actions1, n_actions2 = A.shape
    n_nash = len(set_nash)
    traces, relative_gap = np.array([n_trace]), np.inf

    # if unique equilibrium, is harsanyi selten
    if n_nash == 1:
        return np.ones(1, dtype=bool), traces, relative_gap

    # compute nash equilibria and traces
    trset = gambit_game.tracing(n_trace)
    # extract nash equilibria and traces
    gambit_nash = np.array(list(trset.keys()))
    # compute relative gap (trace computations with small relative gap are ambiguous)
    traces_gambit = np.array(list(trset.values()), dtype=np.int64)
    if len(traces_gambit) > 1:
        traces_sorted = np.sort(traces_gambit)[::-1]
        relative_gap = (traces_sorted[0] - traces_sorted[1]) / traces_sorted[1]

    # relate nash equilibria in gambit_nash to _closest_ nash equilibrium in set_nash
    flattened_set_nash = set_nash.reshape(n_nash, n_actions1 + n_actions2)
    gambit_nash_expanded = gambit_nash[:, np.newaxis, :]
    distances = np.sum(np.abs(flattened_set_nash - gambit_nash_expanded), axis=2)
    closest_idxs = np.argmin(distances, axis=1)
    # relate traces (0 for non-traced equilibria)
    traces = np.zeros(n_nash,dtype=np.int64)
    traces[closest_idxs] = traces_gambit
    # generate harsanyi selten mask (true on maximum trace)
    harsanyi_selten_mask = traces == traces.max()

    return harsanyi_selten_mask, traces, relative_gap
