import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm as tqdm
from tqdm.contrib.concurrent import thread_map

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".pkl")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]
METRIC = "rank"


def effective_rank(output_matrix: np.ndarray) -> float:
    s = np.linalg.svd(output_matrix, compute_uv=False)
    total = s.sum()
    if total == 0:
        return 0.0
    s /= total
    return np.exp(-np.sum(s * np.log(s + 1e-12)))


def compute_rank(
    N: int,
    n: int,
    trial: int,
    use_xp=True,
    input_rank_1=True,
    max_num_samples=250,
    num_washout=10,
):
    pp = ParametricProcess("ktp_780nm_pdc")
    reservoir = PumpShapingProtocol(N, n, pp, use_xp_observables=use_xp)
    reservoir.reset(seed=trial)

    num_samples = min(max_num_samples, num_washout + reservoir.output_dimension)
    if input_rank_1:
        inputs = np.tile(2 * np.random.rand(max_num_samples, 1) - 1, N)
    else:
        inputs = 2 * np.random.rand(max_num_samples, N) - 1

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

    saved_vars = [
        "N",
        "n",
        "trial",
        "rank",
        "num_observables",
        "input_rank_1",
        "use_xp",
    ]
    d = {}
    for name in saved_vars:
        d[name] = locals()[name]
    return d
    return rank, e_rank, reservoir.output_dimension


def compute_results(N_values, num_trials: int):
    results = []
    parameters = [
        (N, n, trial, use_xp, input_rank_1)
        for N in N_values
        for n in range(1, 11)
        for trial in range(num_trials)
        for use_xp in [False, True]
        for input_rank_1 in [False, True]
    ]
    results = thread_map(
        lambda args: compute_rank(*args),
        parameters,
        desc="Parameters and trials",
        leave=False,
        max_workers=4,
    )
    df = pd.DataFrame(results)
    df.to_pickle(RESULTS_FILE)


def plot_reservoir_expressivity(df: pd.DataFrame, ax: plt.Axes):
    N_values = sorted(df["N"].unique())
    custom_colors = plt.cm.managua(np.linspace(0.15, 0.95, len(N_values)))
    for i, N in enumerate(N_values):
        group = df[df["N"] == N].groupby("n")[METRIC]
        mean_ranks = group.mean()
        std_ranks = group.std(ddof=0)
        ax.errorbar(
            mean_ranks.index,
            mean_ranks.values,
            yerr=std_ranks.values,
            label=f"N={N}",
            capsize=5,
            fmt="o-",
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
    ax.grid(True)
    ax.set_xticks(np.arange(2, 11, 2))
    ax.legend(fontsize=9)


def plot_results(df: pd.DataFrame):
    plt.rcParams.update({"font.size": 9})
    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(2, 2)
    for i, use_xp in enumerate([False, True]):
        for j, input_rank_1 in enumerate([False, True]):
            df_subset = df[
                (df["use_xp"] == use_xp) & (df["input_rank_1"] == input_rank_1)
            ]
            plot_reservoir_expressivity(df_subset, axs[i, j])
            xp_label = "$\\hat{q}, \\hat{p}$" if use_xp else "$\\hat{q}$"
            input_rank_label = "rank 1 input" if input_rank_1 else "full rank input"
            axs[i, j].set_title(f"{xp_label} observables, {input_rank_label}")
    fig.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = False

    num_trials = 25
    N_values = [1, 2, 3, 5]

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(N_values, num_trials)

    data = pd.read_pickle(RESULTS_FILE)
    data = data[data["N"].isin(N_values)]
    plot_results(data)
