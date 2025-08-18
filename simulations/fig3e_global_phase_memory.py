from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.task.benchmarks import delay
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
FIGURE_FILES = [
    os.path.join(f"figures/{ext}", FILENAME + "." + ext) for ext in ["png", "pdf"]
]


RESERVOIR_PARAMETERS = ReservoirParameters(
    a=1.65832624e00,
    b=-2.12645646e-04,
    c=-7.22707129e-02,
    d=-1.19260917e00,
    s=0.8,
    v_pi_2=0.075,
)


ALPHA_MASK_PARAMS = {
    1: dict(alpha=-0.25, mask=[0.87]),
    3: dict(
        alpha=[0.25, 0.34, -0.32],
        mask=[[0.33, -0.02, -0.34], [-0.5, -0.28, 0.29], [-0.4, -0.48, -0.35]],
    ),
    5: dict(
        alpha=[-0.17, -0.37, -0.31, -0.14, -0.3],
        mask=[
            [-0.39, 0.16, -0.34, 0.17, 0.32],
            [0.45, -0.21, 0.13, 0.43, 0.32],
            [-0.11, 0.5, 0.29, 0.45, 0.06],
            [0.32, 0.23, -0.06, -0.47, 0.47],
            [-0.26, -0.01, -0.17, -0.12, 0.33],
        ],
    ),
}


def compute_results(params_dicts: dict, noise_level: float, num_trials: int):
    results = []
    saved_vars = ["R", "tau", "capacity"]

    for params in tqdm(params_dicts, desc="Parameter sets"):
        R, tau = params["R"], params["tau"]
        reservoir = GlobalPhaseProtocol(R, RESERVOIR_PARAMETERS)
        for trial in tqdm(range(num_trials), desc="Trial", leave=False):
            alpha_mask_params = ALPHA_MASK_PARAMS[R]
            reservoir.reset(
                noise_level=noise_level,
                alpha=np.array(alpha_mask_params["alpha"]),
                beta=0.0,
                fb_mask=np.array(alpha_mask_params["mask"]),
                noise_seed=trial,
            )

            task = RegressionTask(binary=False)
            task.run(
                reservoir, num_washout=3, num_train=70, num_test=30, seed=2025 + trial
            )
            task.train(partial(delay, delay=tau))
            capacity = task.score().corrcoeff
            # results.append({name: locals()[name] for name in saved_vars})
            d = {}
            for name in saved_vars:
                d[name] = locals()[name]
            results.append(d)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)


def plot_results(df: pd.DataFrame, params_dicts: dict):
    grouped = df.groupby(["R", "tau"])["capacity"].agg(["mean", "std"]).reset_index()
    R_values = grouped["R"].unique()
    tau_values = grouped["tau"].unique()

    fig = plt.figure(figsize=(4, 5))
    colors = plt.cm.managua(np.linspace(0.15, 0.95, len(R_values)))

    for i, R in enumerate(R_values):
        data = grouped[grouped["R"] == R]
        plt.errorbar(
            data["tau"].values,
            data["mean"].values,
            yerr=data["std"].values,
            fmt="-s",
            color=colors[i],
            capsize=4,
            label=f"R={R}",
        )
    plt.ylim([0.0, 1.0])
    plt.xticks(np.array(tau_values), [f"$\\tau={t}$" for t in tau_values])
    plt.text(
        2,
        0.95,
        "Simulated",
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=1),
    )

    plt.ylabel("Memory capacity (Test set)")
    plt.grid(True, axis="y")

    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(R_values),
        frameon=False,
        fontsize=12,
    )
    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    noise_level = 0.05650294519888824
    R_values = [1, 3, 5]
    tau_values = range(4)
    params_dicts = [dict(R=r, tau=t) for r in R_values for t in tau_values]
    num_trials = 30

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(params_dicts, noise_level, num_trials)

    plot_results(pd.read_csv(RESULTS_FILE), params_dicts)
