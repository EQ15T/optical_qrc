import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".pkl")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]


def compute_ranks_and_correlations(
    N: int,
    n: int,
    trial: int,
    max_num_samples=250,
    num_washout=10,
    use_xp=False,
    input_rank_1=False,
):
    pp = ParametricProcess("ktp_780nm_pdc")
    # reservoir = EchoStateNetwork(-N, n)
    reservoir = PumpShapingProtocol(N, n, pp, use_xp_observables=use_xp)
    reservoir.reset(seed=trial)

    num_samples = min(max_num_samples, num_washout + reservoir.output_dimension)
    rng = np.random.RandomState(trial)
    if input_rank_1:
        inputs = np.tile(2 * rng.rand(num_samples, 1) - 1, N)
    else:
        inputs = 2 * rng.rand(num_samples, N) - 1

    observables = np.zeros((num_samples, reservoir.output_dimension))
    for i in tqdm(range(num_samples), desc="Reservoir simulation", leave=False):
        observables[i, :] = reservoir.step(inputs[i, :])

    scaler = StandardScaler()
    observables = scaler.fit_transform(observables)
    last_obs = observables[-reservoir.output_dimension :, :]
    rank = np.linalg.matrix_rank(last_obs, tol=1e-4)

    # Deal gracefully with null series
    if np.any(np.std(last_obs.T, axis=1) == 0):
        corrcoeff = np.full((last_obs.shape[1], last_obs.shape[1]), 0.0)
    else:
        corrcoeff = np.corrcoef(last_obs.T)
    num_observables = reservoir.output_dimension

    saved_vars = ["N", "n", "trial", "rank", "corrcoeff", "num_observables"]
    d = {}
    for name in saved_vars:
        d[name] = locals()[name]
    return d


def compute_results(N_values, num_trials: int):
    parameters = [
        (N, n, trial)
        for N in N_values
        for n in range(1, 11)
        for trial in range(num_trials)
    ]
    results = thread_map(
        lambda args: compute_ranks_and_correlations(*args),
        parameters,
        desc="Parameters and trials",
        leave=False,
        max_workers=1,
    )
    # results = [compute_ranks_and_correlations(*p) for p in parameters]
    df = pd.DataFrame(results)
    df.to_pickle(RESULTS_FILE)


def plot_reservoir_expressivity(df: pd.DataFrame, ax: plt.Axes):
    N_values = sorted(df["N"].unique())
    custom_colors = plt.cm.managua(np.linspace(0.15, 0.95, len(N_values)))

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
            linewidth=0.9,
            color=custom_colors[i],
        )
        if i == 0:
            group = df[df["N"] == N].groupby("n")["num_observables"]
            ax.plot(
                group.mean(),
                marker="+",
                linestyle="dashed",
                color="grey",
                label="Saturation",
            )
    ax.set_xlabel("n")
    ax.set_ylabel("Kernel quality")
    ax.set_title("Reservoir expressivity")
    ax.grid(True)
    ax.set_xticks(np.arange(2, 11, 2))
    ax.legend(fontsize=9, loc="upper left")


def plot_correlation_matrices(df, N_values, axes, n=4, trial=0):
    for i, N in enumerate(N_values):
        entry = df.query(f"N == {N} and n == {n} and trial == {trial}")
        if entry.empty:
            raise ValueError(f"No entry found for N={N}, n={n}, trial={trial}")
        corr_matrix = entry["corrcoeff"].values[0]
        im = axes[i].imshow(
            corr_matrix, cmap="managua", interpolation="nearest", vmin=-1, vmax=1
        )
        axes[i].set_title(f"N={N}, n={n}")
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_xlabel("$\\sigma_{qq}$")
        axes[i].set_ylabel("$\\sigma_{qq}$")
    return im


def plot_results(df: pd.DataFrame):
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(10, 3))

    gs = gridspec.GridSpec(1, 4, width_ratios=[1.1, 1, 1, 0.05], wspace=0.25)
    subplots = [plt.subplot(gs[i]) for i in range(4)]

    plot_reservoir_expressivity(df, subplots[0])

    im = plot_correlation_matrices(df, [1, 5], subplots[1:3], n=6, trial=0)

    cbar = fig.colorbar(im, cax=subplots[3])
    cbar.set_label("Correlation", fontsize=9)
    plt.subplots_adjust(bottom=0.2)
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    num_trials = 25
    N_values = [1, 2, 3, 5, 7]

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(N_values, num_trials)

    data = pd.read_pickle(RESULTS_FILE)

    max_N = 5
    data = data[data["N"] <= max_N]
    plot_results(data)
