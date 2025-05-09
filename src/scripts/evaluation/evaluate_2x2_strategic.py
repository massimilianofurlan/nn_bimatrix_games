import argparse
import os
import torch
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import seaborn as sns

from src.utilities.model_utils import select_models
from src.utilities.training_utils import transpose_game
from src.utilities.viz_utils import set_size
from src.modules.loss_function import Loss


parser = argparse.ArgumentParser(description="Evaluate a model on a dataset of games")
parser.add_argument('--model', type=str, default=None, help="Model folder")
# Process configs
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(1)

# load simulation_metadata
os.system('cls' if os.name == 'nt' else 'clear')
model1, model2, simulation_metadata, model_dir = select_models(model_dir = args.model, device=device)
model1.eval()
model2.eval()

n_players, n_actions = 2, 2


def generate_nash_subspace(n_points, device='cpu'):
    # strategic portion
    set_x = torch.linspace(0, 2 * torch.pi, n_points, device=device)
    set_y = torch.linspace(0, 2 * torch.pi, n_points, device=device)
    X, Y = torch.meshgrid(set_x, set_y, indexing='ij')
    # behavioral portion
    #z in [0,2pi]
    #cos_z = torch.cos(torch.tensor(z, device=device))
    #sin_z = torch.sin(torch.tensor(z, device=device))
    # generate subspace
    A = torch.zeros((n_points, n_points, n_actions, n_actions), device=device)
    B = torch.zeros((n_points, n_points, n_actions, n_actions), device=device)
    # fill in A and B based on the given relations
    A[..., 0, 0] = torch.cos(X)       # +cos_z
    A[..., 0, 1] = torch.sin(X)       # -cos_z
    A[..., 1, 0] = -torch.cos(X)      # +cos_z
    A[..., 1, 1] = -torch.sin(X)      # -cos_z
    B[..., 0, 0] = torch.cos(Y)       # +sin_z
    B[..., 0, 1] = -torch.cos(Y)      # +sin_z
    B[..., 1, 0] = torch.sin(Y)       # -sin_z
    B[..., 1, 1] = -torch.sin(Y)      # -sin_z
    # stack matrices A and B and reshape
    G = torch.stack((A, B), dim=2).view(n_points*n_points, 2, n_actions, n_actions)
    return G

n_points = 301

G = generate_nash_subspace(n_points, device=device)
G_transpose = transpose_game(G)

p = model1(G)
q = model2(G_transpose)

regret1 = Loss.regret(G, p, q)           / (G[:,0].amax(dim=(1,2)) - G[:,0].amin(dim=(1,2)))
regret2 = Loss.regret(G_transpose, q, p) / (G[:,1].amax(dim=(1,2)) - G[:,1].amin(dim=(1,2)))
epsilon_distance_nash = torch.max(regret1,regret2)

p_mesh = p.view(n_points, n_points, n_actions)[:, :, 0].detach().cpu().numpy()
q_mesh = q.view(n_points, n_points, n_actions)[:, :, 0].detach().cpu().numpy()
#regret1_mesh = regret1.view(n_points, n_points).detach().cpu().numpy() 
#regret2_mesh = regret2.view(n_points, n_points).detach().cpu().numpy()
epsilon_distance_nash = regret2.view(n_points, n_points).detach().cpu().numpy()

# --- Font + LaTeX style ---
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 11,
    "font.size": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": 'black',
    'axes.linewidth': 0.5
})

# --- Setup figure and layout ---
fig_width, fig_height = set_size(452.9679, fraction=1)
fig = plt.figure(figsize=(fig_width, fig_height))

# Grid layout: σ1 | σ2 | cbar1 | ε | cbar2
gs = gridspec.GridSpec(
    1, 6,
    width_ratios=[1, 1, 0.05, 0.2, 1, 0.05],
    wspace=0.3
)

# Axes
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharey=ax1)
cbar_ax1 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[4], sharey=ax1)
cbar_ax2 = plt.subplot(gs[5])

# --- Tick positions and labels ---
tick_positions = np.linspace(0, n_points - 1, 5, dtype=int)
tick_labels = [r"$0$", r"", r"$\pi$", r"", r"$2\pi$"]

# --- Heatmaps ---
sns.heatmap(p_mesh.T[::-1, :], cmap="Greys", vmin=0, vmax=1, ax=ax1,
            cbar=False, square=True,
            xticklabels=tick_labels, yticklabels=tick_labels)

sns.heatmap(q_mesh.T[::-1, :], cmap="Greys", vmin=0, vmax=1, ax=ax2,
            cbar=True, cbar_ax=cbar_ax1, square=True,
            xticklabels=tick_labels, yticklabels=False)

sns.heatmap(epsilon_distance_nash.T[::-1, :], cmap="Greys", vmin=0, vmax=0.04, ax=ax3,
            cbar=True, cbar_ax=cbar_ax2, square=True,
            xticklabels=tick_labels, yticklabels=False)

# --- Titles ---
ax1.set_title(r"$\sigma^1$", fontsize=9, fontweight='normal')
ax2.set_title(r"$\sigma^2$", fontsize=9, fontweight='normal')
ax3.set_title(r"$\mathrm{MaxReg}$", fontsize=9, fontweight='normal')

# --- Format axes ---
for ax in [ax1, ax2, ax3]:
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0)
    ax.set_yticks(tick_positions[::-1])
    ax.set_yticklabels(tick_labels)
    ax.tick_params(direction="in", color='grey', width=0.25)
    ax.grid(visible=True, color='grey', linestyle='-', linewidth=0.25, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

# Hide y-tick labels only on ax2 and ax4
ax2.tick_params(labelleft=False)

# --- Style colorbars ---
for cbar_ax, plot_ax in zip([cbar_ax1, cbar_ax2], [ax2, ax3]):
    cbar = plot_ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9, direction='out', length=3, width=0.5, pad=1)
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(0.5)
    
    # Match height to the heatmaps
    plot_pos = plot_ax.get_position()
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([
        cbar_pos.x0-0.01,
        plot_pos.y0,
        cbar_pos.width * 2,
        plot_pos.height
    ])


# --- Export ---
plt.savefig(f"models/{model_dir}/strategic_subspace.pdf", bbox_inches='tight', dpi=300)
print(f'Figure saved at models/{model_dir}/strategic_subspace.pdf')
