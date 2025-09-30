import copy
import enum
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import scipy as sp
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from tqdm.contrib.concurrent import process_map

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir import delay_line_reservoir as dl
from qrc.reservoir import echo_state_network as esn
from qrc.reservoir import long_short_term_memory as lstm

# from qrc.reservoir import opa_feedback_reservoir as opa
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.task.benchmarks import lorenz63, doublescroll
from qrc.task.dynamical_system_task import DynamicalSystemTask

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]
TABLE_FILE = os.path.join(f"results", FILENAME)

# Define all parameters
PARAMETERS = {
    "Lorenz": {
        "best_ESN": {
            "alpha": 0.7,
            "sparsity": 0.05,
            "spectral_radius": 0.8,
            "input_scale": 0.1,
        },
        "worst_ESN": {
            "alpha": 1.0,
            "sparsity": 0.05,
            "spectral_radius": 0.8,
            "input_scale": 5.0,
        },
        "best_LSTM": {
            "epochs": 400,
            "learning_rate": 0.01,
            "dropout_rate": 0.0,
            "seed": 42,
            "show_progress": False,
        },
        "worst_LSTM": {
            "epochs": 400,
            "learning_rate": 0.0001,
            "dropout_rate": 0.0,
            "seed": 42,
            "show_progress": False,
        },
    },
    "Double-scroll": {
        "best_ESN": {
            "alpha": 0.7,
            "sparsity": 0.05,
            "spectral_radius": 0.8,
            "input_scale": 0.5,
        },
        "worst_ESN": {
            "alpha": 1.0,
            "sparsity": 0.15,
            "spectral_radius": 0.8,
            "input_scale": 5.0,
        },
        "best_LSTM": {
            "epochs": 500,
            "learning_rate": 0.02,
            "dropout_rate": 0.0,
            "seed": 42,
            "show_progress": False,
        },
        "worst_LSTM": {
            "epochs": 500,
            "learning_rate": 0.0001,
            "dropout_rate": 0.0,
            "seed": 42,
            "show_progress": False,
        },
    },
}

def compute_single_trial(dataset, label, num_modes, r, trial):
    saved_vars = [
        "label",
        "trial",
        "nmse",
        "num_modes",
        "dimension",
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
    dimension = r.output_dimension

    d = {}
    for name in saved_vars:
        d[name] = locals()[name]

    return d

def trial_wrapper(args):
    dataset, label, num_modes, r, trial = args
    return compute_single_trial(dataset, label, num_modes, r, trial)

def compute_results(taskName, reservoirs: list, num_trials: int, n_jobs=1):
    results = []
    datasets = [taskName]
    for dataset in tqdm(datasets, desc="Dataset"):
        tasks = []
        for label, num_modes, r in reservoirs:
            
            name = None
            if isinstance(r, lstm.LongShortTermMemory):
                config = r.get_config()
                name = "LSTM"
           
            print(label)
            for trial in range(num_trials):
                if name == "LSTM":
                    # Create a new instance of the reservoir for each trial
                    # Deepcopy doesn't work well with tensorflow models
                    new_r = lstm.LongShortTermMemory(**config)
                    tasks.append((dataset, label, num_modes, new_r, trial))
                else:
                    tasks.append((dataset, label, num_modes, copy.deepcopy(r), trial))

        results.extend(
            thread_map(
                trial_wrapper,
                tasks,
                max_workers=n_jobs,
                desc="Reservoirs/trials",
                chunksize=1,
                leave=False,
            )
        )
    df = pd.DataFrame(results)

    path_file = RESULTS_FILE.replace(".csv", f"_{taskName}.csv")
    print(RESULTS_FILE)
    print(f"Saving results to {path_file}")
    df.to_csv(
        path_file,
        mode="a" if os.path.exists(path_file) else "w",   # append if file exists, else write new
        header=not os.path.exists(path_file),             # write header only for new file
        index=False
    )

def LSTM_params(input_dim, hidden_dim, output_dim):
    lstm_layer = 4 * hidden_dim * (hidden_dim + input_dim + 1)
    dense_layer = (hidden_dim + 1) * output_dim
    total = lstm_layer + dense_layer
    return total

def plot_result_2(df, ax, metric, colors, x_var="num_modes", xlabel="Number of modes/neurons"):
    handles = []

    # --- Quantum models: keep boxplots ---
    for name, sub in df.groupby("label"):
        if "General" in name:  # Quantum case
            # x_vals = sorted(sub[x_var].unique())
            label = "General Encoding"
            grouped = sub.groupby(x_var)[metric].agg(
                median="median",
                q25=lambda x: x.quantile(0.25),
                q75=lambda x: x.quantile(0.75),
            )

            x_vals = np.array(grouped.index)
            med = grouped["median"].values
            q25 = grouped["q25"].values
            q75 = grouped["q75"].values

             # Median line
            ax.scatter(
                x_vals, med,
                color=colors[label],
                lw=2,
                label=label,
            )

            # Whisker-like error bars
            yerr = np.vstack([med - q25, q75 - med])
            ax.errorbar(
                x_vals, med, yerr=yerr,
                fmt="none",
                ecolor=colors[label],
                elinewidth=1.2,
                capsize=4,
            )
            
            """ 
            bp = ax.boxplot(
                [sub.loc[sub[x_var] == val, metric].values for val in x_vals],
                positions=x_vals,
                widths=[max(x_vals) * 0.03] * len(x_vals),
                patch_artist=True,
                boxprops=dict(facecolor=colors[name], alpha=0.5, color=colors[name]),
                medianprops=dict(color="black"),
                whiskerprops=dict(color=colors[name]),
                capprops=dict(color=colors[name]),
                flierprops=dict(marker="o", alpha=0.3, markeredgecolor=colors[name]),
            )
            """
            # handles.append(Patch(facecolor=colors[name], edgecolor=colors[name], alpha=0.5, label=name))
            # simplified label
            simple_label = "General Encoding"
            if simple_label not in [h.get_label() for h in handles]:
                 handles.append(Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=colors[name],
                                  markersize=8,
                                  alpha=0.6,
                                  label="Quantum"))


    # --- ESN and LSTM: fill between best/worst ---
    for model in ["ESN", "LSTM"]:
        for fairness in ["unfair", "fair"]:

            qlow = 0.0
            qHigh = 0.0
            for quality in ["best", "worst"]:  # treat separately
                if model == "ESN":
                    label = f"{quality} {model} {fairness}"
                else:
                    label = f"{quality} {model}"

                if label not in df["label"].unique():
                    continue

                grouped = df[df["label"] == label].groupby(x_var)[metric].agg(
                    median="median",
                    q25=lambda x: x.quantile(0.25),
                    q75=lambda x: x.quantile(0.75),
                )

                x_vals = np.array(grouped.index)
                med = grouped["median"].values
                q25 = grouped["q25"].values
                q75 = grouped["q75"].values
                if quality == "best":
                    qlow = q25
                if quality == "worst":
                    qHigh = q75    
                # Median line
                ax.plot(
                    x_vals, med,
                    color=colors[label],
                    lw=2,
                    label=label,
                )

                # Whisker-like error bars
                yerr = np.vstack([med - q25, q75 - med])
                ax.errorbar(
                    x_vals, med, yerr=yerr,
                    fmt="none",
                    ecolor=colors[label],
                    elinewidth=1.2,
                    capsize=4,
                )
        
            ax.fill_between(
                        x_vals, qlow, qHigh,
                        color=colors[label],
                        alpha=0.15,
                    )

         # simplified label for legend
        simple_label = model
        if simple_label not in [h.get_label() for h in handles]:
            handles.append(Patch(facecolor=colors[label], alpha=0.25, label=simple_label))

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)

    return handles

def plot_result(df, ax, metric, colors, x_var="num_modes", xlabel="Number of modes/neurons", mode="boxplot"):
    handles = []
    for name, sub in df.groupby("label"):
        x_vals = sorted(sub[x_var].unique())
        # --- Your original styling logic ---
        if "ESN" in name:
            boxprops = dict(facecolor=colors[name], alpha=0.5, hatch="///", color=colors[name])
            medianprops = dict(color="black", linewidth=1.5)
            m = "*"
        elif "LSTM" in name:
            boxprops = dict(facecolor=colors[name], alpha=0.3, color=colors[name], hatch="|")
            medianprops = dict(color=colors[name], linewidth=1.5)
            m = "s"
        else:  # fallback
            boxprops = dict(facecolor=colors[name], alpha=0.5, color=colors[name])
            medianprops = dict(color="black", linewidth=1.5)
            m = "o"

        if mode == "boxplot":
            # (your existing ESN/LSTM styling logic)
            m = "o"
            bp = ax.boxplot(
                [sub.loc[sub[x_var] == val, metric].values for val in x_vals],
                positions=x_vals,
                widths=[val * 0.1 for val in x_vals],
                patch_artist=True,
                boxprops=boxprops,
                medianprops=dict(color="black"),
                whiskerprops=dict(color=colors[name]),
                capprops=dict(color=colors[name]),
                flierprops=dict(marker=m, alpha=0.3, markeredgecolor=colors[name]),
            )
            # Use a Patch for the legend
            handles.append(
                Patch(facecolor=boxprops.get("facecolor", colors[name]),
                    edgecolor=colors[name],
                    hatch=boxprops.get("hatch", ""),
                    alpha=boxprops.get("alpha", 0.6),
                    label=name)
            )

        elif mode == "std":
            line = ax.errorbar(
                x_vals,
                [sub.loc[sub[x_var] == val, metric].mean() for val in x_vals],
                yerr=[sub.loc[sub[x_var] == val, metric].std() for val in x_vals],
                fmt=m,
                capsize=4,
                label=name,
                color=colors[name],
            )
            handles.append(line[0])  # errorbar returns (line, caplines, barlinecols)

        elif mode == "trials":
            sc = ax.scatter(
                sub[x_var],
                sub[metric],
                alpha=0.6,
                label=name,
                color=colors[name],
                marker=m,
            )
            handles.append(sc)

    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    return handles

def plot_results(df, taskName):
    # Explicit groups
    quantum_labels = ["General Encoding"]
    lstm_labels = ["best LSTM", "worst LSTM"]
    unfair_esn_labels = ["best ESN unfair", "worst ESN unfair"]
    fair_esn_labels   = ["best ESN fair", "worst ESN fair"]

    # Label sets per plot
    unfair_labels = lstm_labels + unfair_esn_labels +  quantum_labels 
    fair_labels   = fair_esn_labels + quantum_labels 

    # Fixed colors across both panels
    color_map = {
        "General Encoding": "C1",
        "best LSTM": "C0",
        "worst LSTM": "C0",
        "best ESN unfair": "C3",
        "worst ESN unfair": "C3",
        "best ESN fair": "C3",   # same as best ESN unfair
        "worst ESN fair": "C3",  # same as worst ESN unfair
    }

    metric = "nmse"
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # --- Left: Unfair ---
    subdf_unfair = df[df["label"].isin(unfair_labels)]
    
    ax = fig.add_subplot(gs[0, 0])
    # handles_unfair =plot_result(subdf_unfair, ax, metric, color_map, xlabel="modes/neurons/hidden_units", mode="boxplot")
    handles_unfair = plot_result_2(subdf_unfair, ax, metric, color_map, xlabel="modes/neurons/hidden_units")

    ax.set_title("Same amount of neurons for the \n ESN and Quantum models (''Unfair'')")
    ax.set_ylabel("NMSE")
    ax.set_ylim(1e-6, 10.0)
    # legend (manual)
    """ 
    legend_handles = []
    for label in unfair_labels:
        if "General" in label:
            # Star marker
            legend_handles.append(
                Patch(facecolor=color_map[label], edgecolor=color_map[label], alpha=0.5, label=label)
            )
        elif "ESN" in label:
            # Filled box style for legend
            legend_handles.append(
                Patch(facecolor=color_map[label], edgecolor=color_map[label], hatch="///", alpha=0.8, label=label)
            )
        elif "LSTM" in label:
            # Hollow box with hatch
            legend_handles.append(
                Patch(facecolor="white", edgecolor=color_map[label], hatch="//", label=label)
            )
    """
    ax.legend(handles=handles_unfair , frameon=True, loc="lower left")

    # --- Right: Fair ---
    subdf_fair = df[df["label"].isin(fair_labels)].copy()
    quantumMask= subdf_fair["label"] == "General Encoding"
    subdf_fair.loc[quantumMask, "num_modes"] = (
        subdf_fair.loc[quantumMask, "num_modes"] * (subdf_fair.loc[quantumMask, "num_modes"] + 1) // 2
)
    #lstmMask = subdf_fair["label"].str.contains("LSTM")
    # subdf_fair.loc[lstmMask, "num_modes"] = LSTM_params(3, subdf_fair.loc[lstmMask, "num_modes"], 3)  # 3 input features

    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(1e-6, 10.0)
    ax.set_xlim(6, 110)
    # handles_fair = plot_result(subdf_fair, ax, metric, color_map, xlabel="Amount of parameters", mode="boxplot")
    handles_fair = plot_result_2(subdf_fair, ax, metric, color_map, xlabel="Amount of parameters")
    ax.set_title("Same amount of parameters for the \nESN and Quantum models (''Fair'')")
    ax.set_ylabel("NMSE")
       
    # legend (manual)
    # legend (manual)
    """
    legend_handles = []
    for label in fair_labels:
        if "General" in label:
            # Star marker
            legend_handles.append(
                Patch(facecolor=color_map[label], edgecolor=color_map[label], alpha=0.5, label=label)
            )
        elif "ESN" in label:
            # Filled box style for legend
            legend_handles.append(
                Patch(facecolor=color_map[label], edgecolor=color_map[label], hatch="///", alpha=0.8, label=label)
            )
    """
    
    # ax.legend(handles=handles_fair, frameon=True)
    
    fig.suptitle(f"One-step-ahead prediction ({taskName})", fontsize=20) 
    fig.tight_layout()
    for f in FIGURE_FILES:
        f = f.replace(".png", f"_{taskName}.png").replace(".pdf", f"_{taskName}.pdf")
        print(f"Saving figure to {f}")
        plt.savefig(f)
    return fig

if __name__ == "__main__":
    taskName= sys.argv[1]
    print("Task name:", taskName)
    force_run = False
    num_trials = 25
    pp = ParametricProcess("ktp_780nm_pdc", 180)

    quantum_reservoirs = []
    bestESN_unfair = [] # Same neurons as quantum modes (not same amount of parameters)
    bestESN_fair = [] # Same neurons as quantum observables (same amount of parameters)
    worstESN_unfair = [] # Same neurons as quantum modes (not same amount of parameters)
    worstESN_fair = [] # Same neurons as quantum observables (same amount of parameters)
    bestLSTM = []
    worstLSTM = []

   # Access parameters
    best_ESN_params = PARAMETERS[taskName]["best_ESN"]
    worst_ESN_params = PARAMETERS[taskName]["worst_ESN"]
    best_LSTM_params = PARAMETERS[taskName]["best_LSTM"]
    worst_LSTM_params = PARAMETERS[taskName]["worst_LSTM"]

    print("Best ESN params:", best_ESN_params)
    print("Worst ESN params:", worst_ESN_params)
    print("Best LSTM params:", best_LSTM_params)
    print("Worst LSTM params:", worst_LSTM_params)

    n_hidden_lstm = [1, 3, 5, 7, 9, 11]
    for i, n in enumerate([4, 6, 8, 10, 12, 14]):
        equiv_neurons = n * (n + 1) // 2
        
        quantum_reservoirs.append(
            (
                "General Encoding",
                n,
                PumpShapingProtocol(15, n, pp, alpha_scale=0.5, feedback_scale=0.3),
            )
        )

        bestESN_unfair.append(
            (
                "best ESN unfair",
                n, # Same neurons as quantum modes (not same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    n,
                    **best_ESN_params,
                ),
            )
        )

        bestESN_fair.append(
            (
                "best ESN fair",
                equiv_neurons, # Same neurons as quantum observables (same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    equiv_neurons,
                    **best_ESN_params,
                ),
            )
        )

        worstESN_unfair.append(
            (
                "worst ESN unfair",
                n, # Same neurons as quantum modes (not same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    n,
                    **worst_ESN_params,
                ),
            )
        )

        worstESN_fair.append(
            (
                "worst ESN fair",
                equiv_neurons, # Same neurons as quantum observables (same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    equiv_neurons,
                    **worst_ESN_params,
                ),
            )
        )

        hidden_dim = n_hidden_lstm[i]
        bestLSTM.append(
            (
                "best LSTM",
                hidden_dim,
                lstm.LongShortTermMemory(
                    input_dim=3,
                    out_dim=3,
                    hidden_dim=hidden_dim,
                    **best_LSTM_params,
                )
            )
        )

        worstLSTM.append(
            (
                "worst LSTM",
                hidden_dim,
                lstm.LongShortTermMemory(
                    input_dim=3,
                    out_dim=3,
                    hidden_dim=hidden_dim,
                    **worst_LSTM_params,
                )
            )
        )

    path_file = RESULTS_FILE.replace(".csv", f"_{taskName}.csv")
    if force_run or not os.path.exists(path_file):
        print("Computting quantum reservoir results...", len(quantum_reservoirs))
        compute_results(taskName,quantum_reservoirs, num_trials)

        print("Computting best ESN unfair results...", len(bestESN_unfair))
        compute_results(taskName, bestESN_unfair, num_trials)

        print("Computting best ESN fair results...", len(bestESN_fair))
        compute_results(taskName, bestESN_fair, num_trials)

        print("Computting worst ESN unfair results...", len(worstESN_unfair))
        compute_results(taskName, worstESN_unfair, num_trials)

        print("Computting worst ESN fair results...", len(worstESN_fair))
        compute_results(taskName, worstESN_fair, num_trials)

        print("Computting best LSTM results...", len(bestLSTM))
        compute_results(taskName, bestLSTM, num_trials)

        print("Computting worst LSTM results...", len(worstLSTM))
        compute_results(taskName, worstLSTM, num_trials)

    
    df = pd.read_csv(path_file)
    plot_results(df, taskName)
