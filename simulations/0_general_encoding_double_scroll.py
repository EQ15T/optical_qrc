from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.reservoir import delay_line_reservoir as dl
from qrc.task.benchmarks import doublescroll
from qrc.task.dynamical_system_task import DynamicalSystemTask

from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results", FILENAME + ".csv")
FIGURE_FILES = [os.path.join(f"figures/{ext}", FILENAME + "." + ext) for ext in ['png', 'pdf']]
METRIC = "NMSE"


def compute_results(params_dicts: dict, num_trials: int):
    results = []
    saved_vars = ["reservoir", "N", "n", "trial", "NMSE", "corrcoeff"]
    for params in tqdm(params_dicts, desc="Parameter sets"):
        reservoir = params["reservoir"]
        if reservoir == "pump shaping":
            N, n = params["N"], params["n"]
            pp = ParametricProcess("ktp_780nm_pdc")
            r = PumpShapingProtocol(N, n, pp)
            trials = range(num_trials)
        else:
            N, n = params["tau"], params["degree"]
            r = dl.DelayLineReservoir(3, N, n)
            trials = range(1)

        for trial in tqdm(trials, desc="Trial", leave=False):
            task = DynamicalSystemTask(
                partial(doublescroll, seconds_per_point=1.5), closed_loop=False
            )
            if reservoir == "pump shaping":
                r.reset(seed=trial)
            else:
                r.reset()
            task.run(r, num_washout=10, num_train=250, num_test=50)
            task.train()
            result = task.score()
            NMSE, corrcoeff = result.nmse, result.corrcoeff
            results.append({name: locals()[name] for name in saved_vars})
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)


def plot_results(df):
    # Group by N and n, and compute mean and std of nmse
    grouped = (
        df.groupby(["N", "n", "reservoir"])[METRIC].agg(["mean", "std"]).reset_index()
    )

    labels = []
    for _, row in grouped.iterrows():
        if row.reservoir == "pump shaping":
            label = f"{row.reservoir}\nN={int(row.N)}, n={int(row.n)}"
        else:
            label = f"{row.reservoir}\n$\\tau = {int(row.N)}$, d={int(row.n)}"
        labels.append(label)
    x = range(len(grouped))

    plt.figure(figsize=(6, 3))
    plt.errorbar(x, grouped["mean"], yerr=grouped["std"], fmt="o", capsize=5)
    plt.xticks(range(len(grouped)), labels, rotation=45, ha="right")
    if METRIC == "NMSE":
        plt.yscale("log")
    plt.ylabel(f"{METRIC}")
    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = False

    num_trials = 10
    params_dicts = [
        {"reservoir": "pump shaping", "N": 6, "n": 4},
        {"reservoir": "pump shaping", "N": 9, "n": 4},
        {"reservoir": "pump shaping", "N": 9, "n": 6},
        {"reservoir": "delay line", "tau": 0, "degree": 1},
        {"reservoir": "delay line", "tau": 1, "degree": 1},
    ]

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(params_dicts, num_trials)

    plot_results(pd.read_csv(RESULTS_FILE))
