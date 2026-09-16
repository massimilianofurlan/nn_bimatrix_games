import pytest

from src.modules.sampler import BimatrixSampler


games = BimatrixSampler(payoffs_space='sphere', n_actions=4)(8).numpy()

@pytest.fixture(params=games)
def G(request):
    return request.param
