from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.task.benchmarks import parity_check
from qrc.task.regression_task import RegressionTask

from functools import partial
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
METRIC = "accuracy"


def compute_results(params_dicts: dict, num_trials: int):
    results = []
    saved_vars = ["N", "n", "trial", "tau", "accuracy", "corrcoeff"]

    pp = ParametricProcess("ktp_780nm_pdc")

    for params in tqdm(params_dicts, desc="Parameter sets"):
        N, n = params["N"], params["n"]
        reservoir = PumpShapingProtocol(N, n, pp)

        for trial in tqdm(range(num_trials), desc="Trial", leave=False):
            task = RegressionTask()

            # Pump shaping parameters and random input sequence are seeded
            # by the trial index for reproducible results.
            reservoir.reset(seed=trial)
            task.run(reservoir, num_washout=10, num_train=180, num_test=50, seed=trial)

            for tau in range(6):
                task.train(partial(parity_check, tau=tau))
                result = task.score()
                accuracy, corrcoeff = result.accuracy, result.corrcoeff
                # results.append({name: locals()[name] for name in saved_vars})
                d = {}
                for name in saved_vars:
                    d[name] = locals()[name]
                results.append(d)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)


def mean_std_curve(df_subset: pd.DataFrame, label: str, color: np.ndarray):
    grouped = df_subset.groupby("tau")[METRIC].agg(["mean", "std"]).reset_index()
    plt.errorbar(
        grouped["tau"].values,
        100 * grouped["mean"].values,
        yerr=100 * grouped["std"].values,
        fmt="-s",
        color=color,
        capsize=4,
        label=label,
    )


def plot_results(df: pd.DataFrame, params_dicts: dict):
    plt.figure(figsize=(5, 4))
    colors = plt.cm.managua(np.linspace(0.15, 0.95, len(params_dicts)))

    for i, params in enumerate(params_dicts):
        subset = df[(df["N"] == params["N"]) & (df["n"] == params["n"])]
        label = f"N={params['N']}, n={params['n']}"
        mean_std_curve(subset, label, color=colors[i])

    plt.xlabel("$\\tau$")
    plt.ylabel("Parity check accuracy\non the test set (%)")
    plt.grid(True)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    params_dicts = [
        {"N": 1, "n": 4},
        {"N": 1, "n": 6},
        {"N": 5, "n": 4},
        {"N": 5, "n": 6},
    ]
    num_trials = 10

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(params_dicts, num_trials)

    plot_results(pd.read_csv(RESULTS_FILE), params_dicts)
