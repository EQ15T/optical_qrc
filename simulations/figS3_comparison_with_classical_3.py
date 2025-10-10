import copy
import os

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir import delay_line_reservoir as dl
from qrc.reservoir import echo_state_network as esn

from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.task.benchmarks import doublescroll, lorenz63
from qrc.task.dynamical_system_task import DynamicalSystemTask

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]
TABLE_FILE = os.path.join(f"results", FILENAME)


def compute_single_trial(dataset, label, num_modes, r, trial):
    saved_vars = [
        "label",
        "trial",
        "nmse",
        "num_modes",
        "dimension",
        "psde",
        "psde_emd",
        "dataset",
    ]

    fn = doublescroll if "scroll" in dataset else lorenz63
    t = DynamicalSystemTask(fn)

    if type(r) is dl.DelayLineReservoir:
        r.reset()
    else:
        r.reset(seed=trial)

    t.run(r, num_washout=50, num_train=2500, num_test=250)

    t.train(alpha=0)
    nmse = t.score(free_running=False).nmse

    # Re-train with Ridge regression for stability
    t.train(alpha=1e-4)
    results = t.score(free_running=True)
    psde, psde_emd = results.psde, results.psde_emd

    dimension = r.output_dimension
    d = {}
    for name in saved_vars:
        d[name] = locals()[name]

    return d


def compute_results(reservoirs: list, num_trials: int, n_jobs=4):
    results = []
    datasets = ["Double-scroll", "Lorenz"]
    for dataset in tqdm(datasets, desc="Dataset"):
        tasks = []
        for label, num_modes, r in reservoirs:
            for trial in range(num_trials):
                tasks.append((dataset, label, num_modes, copy.deepcopy(r), trial))

        # Run trials and reservoirs in parallel
        results.extend(
            thread_map(
                lambda args: compute_single_trial(*args),
                tasks,
                max_workers=n_jobs,
                desc="Reservoirs/trials",
                leave=False,
            )
        )
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)


def plot_result(df, ax, metric, colors, x_var="num_modes"):
    darken = lambda color: tuple([v * 0.7 for v in mcolors.to_rgb(color)])
    for name, sub in df.groupby("label"):
        if len(sub) == 1 or "Naive" in name:
            ax.scatter(
                sub[x_var],
                sub[metric],
                marker="*",
                s=24,
                label=name,
                color=colors[name],
            )
        else:
            ax.boxplot(
                [
                    sub.loc[sub[x_var] == val, metric].values
                    for val in sorted(sub[x_var].unique())
                ],
                positions=sorted(sub[x_var].unique()),
                widths=sorted(sub[x_var].unique() * 0.1),
                patch_artist=True,
                boxprops=dict(facecolor=colors[name], alpha=0.5, color=colors[name]),
                medianprops=dict(color=darken(colors[name])),
                whiskerprops=dict(color=colors[name]),
                capprops=dict(color=colors[name]),
                flierprops=dict(
                    marker="o", alpha=0.3, markeredgecolor=colors[name], markersize=5
                ),
            )

    ax.set_xscale("log")
    ax.set_xticks(2 ** np.arange(1, 8), labels=2 ** np.arange(1, 8))

    from matplotlib.ticker import NullLocator

    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Number of modes/neurons")
    ax.set_yscale("log")
    ax.grid(True, which="major", linestyle="--", linewidth=0.5)


def plot_results(df, show_continuation_results=False):
    datasets = df["dataset"].unique()
    names = df["label"].unique()
    # colors = plt.cm.managua(np.linspace(0.0, 1.0, len(names)))
    colors = [f"C{i}" for i in range(len(names))]
    colors = {elem: colors[i] for i, elem in enumerate(names)}
    num_rows = 2 if show_continuation_results else 1
    num_datasets = len(datasets)

    fig = plt.figure(figsize=(5 * num_datasets, 4 * num_rows))
    gs = gridspec.GridSpec(
        num_rows + 1, num_datasets, height_ratios=[1] * num_rows + [0.1], figure=fig
    )

    metrics = ["nmse", "psde_emd"]
    for j, dataset in enumerate(datasets):
        subs_df = df[df["dataset"] == dataset]
        for evaluation_index in range(num_rows):
            ax = fig.add_subplot(gs[evaluation_index, j])
            plot_result(subs_df, ax, metrics[evaluation_index], colors)
            if evaluation_index == 0:
                ax.set_title(f"{dataset.title()} | One-step-ahead prediction")
                ax.set_ylabel("NMSE")
                # ax.set_ylim([0.5e-4, 1.0])
            else:
                ax.set_title(f"{dataset.title()} | Free-running continuation")
                ax.set_ylabel("Spectral distance")

    legend_ax = fig.add_subplot(gs[num_rows, :])
    legend_ax.axis("off")

    handles = [
        mlines.Line2D([], [], color=colors[name], marker="o", linestyle="", label=name)
        for name in colors
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=2,
        frameon=True,
    )

    fig.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = False
    num_trials = 25
    pp = ParametricProcess("ktp_780nm_pdc", 180)
    reservoirs = [("Naive", 3, dl.DelayLineReservoir(3, 0, 1))]

    for n in [4, 6, 8, 10, 12, 14]:
        equiv_neurons = n * (n + 1) // 2
        reservoirs.append(
            (
                "General Encoding",
                n,
                PumpShapingProtocol(15, n, pp, alpha_scale=0.5, feedback_scale=0.3),
            )
        )
        reservoirs.append(
            (
                "Echo State Network",
                equiv_neurons,
                esn.EchoStateNetwork(
                    3,
                    equiv_neurons,
                    alpha=0.5,
                    sparsity=0.0,
                    spectral_radius=0.8,
                    input_scale=0.5,
                ),
            )
        )

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(reservoirs, num_trials)

    df = pd.read_csv(RESULTS_FILE)
    prior_work = pd.DataFrame(
        [
            {
                "label": "Wang (PhysRevRes.6.043183)",
                "trial": 0,
                "nmse": np.mean(np.array([8.9e-3, 11.7e-3, 8.8e-3]) ** 2),
                "dimension": 62,
                "num_modes": 16,
                "dataset": "Double-scroll",
            },
            {
                "label": "Wang (PhysRevRes.6.043183)",
                "trial": 0,
                "nmse": np.mean(np.array([3.7e-3, 14.6e-3, 7.5e-3]) ** 2),
                "dimension": 62,
                "num_modes": 16,
                "dataset": "Lorenz",
            },
        ]
    )
    df = pd.concat([df, prior_work])
    plot_results(df, show_continuation_results=False)
