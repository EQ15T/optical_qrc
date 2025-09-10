import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".pkl")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]


def compute_ranks_and_correlations(inputs, N: int, n: int, trial: int, num_samples=50):
    pp = ParametricProcess("ktp_780nm_pdc")
    reservoir = PumpShapingProtocol(N, n, pp, use_xp_observables=True)
    reservoir.reset()

    observables = np.zeros((num_samples, reservoir.output_dimension))
    for i in tqdm(range(num_samples), desc="Reservoir simulation", leave=False):
        observables[i, :] = reservoir.step(inputs[i, :], gain=1.72693881974)

    scaler = StandardScaler()
    observables = scaler.fit_transform(observables)

    last_obs = observables[-reservoir.output_dimension :, :]
    rank = np.linalg.matrix_rank(last_obs, tol=1e-4)
    corrcoeff = np.corrcoef(last_obs.T)

    # corrcoeff = np.corrcoef(observables.T)
    # rank = np.linalg.matrix_rank(corrcoeff, tol=1e-4)
    return rank, corrcoeff


def compute_results(N_values, num_trials: int):
    results = []
    num_samples = 50
    saved_vars = ["N", "n", "trial", "rank", "corrcoeff"]
    np.random.seed(0)
    for N in tqdm(N_values, desc="N value", leave=True):
        for trial in tqdm(range(num_trials), desc="Trial", leave=False):
            inputs = np.tile(2 * np.random.rand(num_samples, 1), N)
            for n in tqdm(range(1, 10), desc="n value", leave=False):
                rank, corrcoeff = compute_ranks_and_correlations(inputs, N, n, trial)
                # results.append({name: locals()[name] for name in saved_vars})
                d = {}
                for name in saved_vars:
                    d[name] = locals()[name]
                results.append(d)

    df = pd.DataFrame(results)
    df.to_pickle(RESULTS_FILE)


def plot_reservoir_expressivity(df: pd.DataFrame, ax: plt.Axes):
    custom_colors = ["royalblue", "orangered"]
    N_values = sorted(df["N"].unique())

    for i, N in enumerate(N_values):
        group = df[df["N"] == N].groupby("n")["rank"]
        mean_ranks = group.mean()
        std_ranks = group.std(ddof=0)
        ax.errorbar(
            mean_ranks.index,
            mean_ranks.values,
            yerr=std_ranks.values,
            label=f"N={N}",
            capsize=5,
            fmt="o-",
            color=custom_colors[i % 2],
        )
    ax.set_xlabel("n")
    ax.set_ylabel("Rank")
    ax.set_title("Reservoir expressivity")
    ax.grid(True)
    ax.set_xticks(np.arange(2, 10, 2))
    ax.legend(fontsize=9)


def reorder_correlation_matrix(mat, n):
    original_pairs = [(i, j) for i in range(2 * n) for j in range(i, 2 * n)]
    new_ordered_pairs = (
        [(i, j) for i in range(n) for j in range(i, n)]  # x
        + [(i, j) for i in range(n, 2 * n) for j in range(i, 2 * n)]  # p
        + [(i, j) for i in range(n) for j in range(n, 2 * n)]  # xp
    )
    idx = [new_ordered_pairs.index(pair) for pair in original_pairs]
    return mat[np.ix_(idx, idx)]


def plot_reordered_correlation_matrices(df, N_values, axes, n=4, trial=9):
    custom_colors = ["royalblue", "orangered"]

    for i, N in enumerate(N_values):
        entry = df.query(f"N == {N} and n == {n} and trial == {trial}")
        if entry.empty:
            raise ValueError(f"No entry found for N={N}, n={n}, trial={trial}")
        corr_matrix = entry["corrcoeff"].values[0]
        corr_matrix = reorder_correlation_matrix(corr_matrix, n)
        im = axes[i].imshow(
            corr_matrix, cmap="managua", interpolation="nearest", vmin=-1, vmax=1
        )
        axes[i].set_title(f"n={n}, N={N}")
        k = n * (n + 1) // 2
        ticks = [k // 2, 3 * k // 2, 2 * k + n**2 / 2]
        labels = ["$\\sigma_{qq}$", "$\\sigma_{pp}$", "$\\sigma_{qp}$"]

        axes[i].set_xticks(ticks)
        axes[i].set_yticks(ticks)
        axes[i].set_xticklabels(labels, fontsize=7)
        axes[i].set_yticklabels(labels, fontsize=7)
    return im


def plot_results(df: pd.DataFrame, params_dicts: list):
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(10, 3))

    gs = gridspec.GridSpec(1, 4, width_ratios=[1.1, 1, 1, 0.05], wspace=0.25)
    subplots = [plt.subplot(gs[i]) for i in range(4)]

    plot_reservoir_expressivity(df, subplots[0])

    im = plot_reordered_correlation_matrices(df, N_values, subplots[1:3], trial=0)

    cbar = fig.colorbar(im, cax=subplots[3])
    cbar.set_label("Correlation", fontsize=9)
    plt.subplots_adjust(bottom=0.2)
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    num_trials = 5
    N_values = [1, 5]

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(N_values, num_trials)

    data = pd.read_pickle(RESULTS_FILE)
    plot_results(data, N_values)
