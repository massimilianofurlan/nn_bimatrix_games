from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import torch

from src.modules.loss_function import Loss
from src.modules.mlp import MLP_Bimatrix
from src.modules.sampler import BimatrixSampler
from src.utilities.bimatrix_utils import get_maxmin_payoff
from src.utilities.eval_utils import get_value
from src.utilities.io_utils import process_config, save_to_pickle
from src.utilities.model_utils import initialize_scheduler, initialize_weigths
from src.utilities.training_utils import train


def test_maxmin_payoff_sign():
    game = np.array([
        [[1.0, 1.0], [0.0, 0.0]],
        [[2.0, 0.0], [2.0, 0.0]],
    ])
    np.testing.assert_allclose(get_maxmin_payoff(game), [1.0, 2.0])


def test_get_value_preserves_dtype():
    indices = np.array([0, 0, 0])
    assert get_value(indices, [np.array([True]), np.array([False]), np.array([True])]).dtype == bool
    np.testing.assert_array_equal(
        get_value(indices, [np.array([-1]), np.array([0]), np.array([1])]),
        [-1, 0, 1],
    )


def test_sampler_constraints():
    torch.manual_seed(1)
    preferences = BimatrixSampler(2, payoffs_space='sphere_preferences')(512)
    strategic = BimatrixSampler(2, payoffs_space='sphere_strategic')(512)
    zero_sum = BimatrixSampler(2, payoffs_space='sphere', game_class='zero_sum')(512)
    assert torch.allclose(preferences.sum(dim=(2, 3)), torch.zeros(512, 2), atol=1e-5)
    assert torch.allclose(preferences.norm(dim=(2, 3)), torch.full((512, 2), 2.0), atol=1e-5)
    assert torch.allclose(strategic[:, 0].sum(dim=1), torch.zeros(512, 2), atol=1e-5)
    assert torch.allclose(strategic[:, 1].sum(dim=2), torch.zeros(512, 2), atol=1e-5)
    assert torch.allclose(zero_sum[:, 0], -zero_sum[:, 1], atol=1e-5)


def test_short_training():
    torch.manual_seed(1)
    model1 = MLP_Bimatrix(2, 1, 8)
    model2 = MLP_Bimatrix(2, 1, 8)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.01)
    _, _, regrets = train(
        model1,
        optimizer1,
        initialize_scheduler(optimizer1, 1.0),
        model2,
        optimizer2,
        initialize_scheduler(optimizer2, 1.0),
        16,
        1,
        Loss(),
        BimatrixSampler(2),
        '',
    )
    assert regrets.shape == (16, 2)
    assert np.isfinite(regrets).all()


def test_fixed_training_set(tmp_path, monkeypatch):
    dataset_dir = tmp_path / 'data' / 'fixed'
    dataset_dir.mkdir(parents=True)
    dataset = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)
    save_to_pickle(dataset, dataset_dir / 'dataset.pkl')
    monkeypatch.chdir(tmp_path)
    config = {
        'n_actions': 3,
        'bimatrix': {'payoffs_space': 'sphere', 'game_class': 'general_sum'},
        'model1': {},
        'model2': {},
    }
    processed, training_set = process_config(SimpleNamespace(training_set='fixed'), deepcopy(config))
    assert tuple(training_set.shape) == (3, 2, 2, 2)
    assert processed['n_actions'] == 2
    assert processed['bimatrix']['n_actions'] == 2
    assert processed['model1']['n_actions'] == 2
    assert processed['model2']['n_actions'] == 2
    sampled = BimatrixSampler(**processed['bimatrix'], set_games=training_set)(12)
    assert all(any(torch.equal(game, fixed) for fixed in training_set) for game in sampled)


def test_initial_model_weights(tmp_path, monkeypatch):
    model_dir = tmp_path / 'models' / 'initial'
    model_dir.mkdir(parents=True)
    source1 = MLP_Bimatrix(2, 1, 8)
    source2 = MLP_Bimatrix(2, 1, 8)
    torch.save({'model_state_dict': source1.state_dict()}, model_dir / 'model1.pth')
    torch.save({'model_state_dict': source2.state_dict()}, model_dir / 'model2.pth')
    target1 = torch.jit.script(MLP_Bimatrix(2, 1, 8))
    target2 = torch.jit.script(MLP_Bimatrix(2, 1, 8))
    monkeypatch.chdir(tmp_path)
    initialize_weigths(target1, target2, 'initial')
    for source, target in zip(source1.parameters(), target1.parameters()):
        assert torch.equal(source, target)
    for source, target in zip(source2.parameters(), target2.parameters()):
        assert torch.equal(source, target)

