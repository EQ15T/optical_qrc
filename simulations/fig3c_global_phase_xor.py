from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.task.benchmarks import parity_check
from qrc.task.regression_task import RegressionTask

from functools import partial
from matplotlib import gridspec
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results", FILENAME + ".csv")
FIGURE_FILES = [os.path.join(f"figures/{ext}", FILENAME + "." + ext) for ext in ['png', 'pdf']]


RESERVOIR_PARAMETERS = ReservoirParameters(
    a=1.61516986e00,
    b=-2.51162304e-04,
    c=-3.37095412e-01,
    d=-1.15273661e00,
    s=0.8,
    v_pi_2=0.075,
)


def compute_results(params_dicts: dict, num_trials: int):
    results = []
    saved_vars = ["noise", "training_size", "accuracy"]

    for params in tqdm(params_dicts, desc="Parameter sets"):
        noise, training_size = params["noise"], params["training_size"]
        reservoir = GlobalPhaseProtocol(1, RESERVOIR_PARAMETERS)
        for trial in tqdm(range(num_trials), desc="Trial", leave=False):
            task = RegressionTask()
            scale = (np.pi / 2) / RESERVOIR_PARAMETERS.v_pi_2
            reservoir.reset(
                noise_level=noise,
                alpha=np.pi / 2,
                beta=-0.01 * scale,
                fb_mask=np.array([0.035]) * scale,
                noise_seed=trial,
            )
            task.run(
                reservoir,
                num_washout=5,
                num_train=training_size,
                num_test=30,
                seed=trial,
            )
            task.train(partial(parity_check, tau=1))
            accuracy = task.score().accuracy
            results.append({name: locals()[name] for name in saved_vars})

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)


def plot_results(df: pd.DataFrame, params_dicts: dict):
    grouped = (
        df.groupby(["training_size", "noise"])["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )
    noise_levels = grouped["noise"].unique()

    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.05, 0.95], hspace=0.0)
    ax = plt.subplot(gs[1])
    colors = plt.cm.managua(np.linspace(0.15, 0.95, len(noise_levels)))

    for i, noise in enumerate(noise_levels):
        data = grouped[grouped["noise"] == noise]
        ax.errorbar(
            data["training_size"].values,
            100 * data["mean"].values,
            yerr=(100 if i in [0, 3] else 0) * data["std"].values,
            fmt="-s",
            color=colors[i],
            capsize=4,
            label=f"{noise:.3f}",
        )
    ax.set_ylim([50, 100])
    ax.text(
        0.814,
        0.12,
        "Simulated",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=1),
    )

    plt.xlabel("Training size")
    plt.ylabel("XOR accuracy on the test set (%)")
    plt.grid(True)

    # Manually recreate the legend to get the correct layout
    legend_ax = plt.subplot(gs[0])
    legend_ax.axis("off")
    y = 0.9
    x = 0.35
    legend_ax.text(
        x,
        y,
        "Gaussian noise",
        va="center",
        ha="right",
        transform=legend_ax.transAxes,
        fontsize=10,
    )
    for i, noise in enumerate(noise_levels):
        x_pos = i * 0.15 + 0.05 + x
        legend_ax.add_patch(
            Rectangle(
                (x_pos, y - 0.02),
                0.03,
                0.08,
                color=colors[i],
                transform=legend_ax.transAxes,
            )
        )
        legend_ax.text(
            x_pos + 0.04,
            y,
            f"{noise:.3f}",
            va="center",
            ha="left",
            transform=legend_ax.transAxes,
            fontsize=10,
        )

    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    noise_values = [0.058, 0.078, 0.113, 0.183]
    training_size_values = range(10, 100, 10)
    params_dicts = [
        dict(noise=n, training_size=t)
        for n in noise_values
        for t in training_size_values
    ]
    num_trials = 100

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(params_dicts, num_trials)

    plot_results(pd.read_csv(RESULTS_FILE), params_dicts)
