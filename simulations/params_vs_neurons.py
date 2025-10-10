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

# Define optimized parameters
PARAMETERS = {
    "Lorenz": {
        "best_ESN": {
            "alpha": 0.7,
            "sparsity": 0.05,
            "spectral_radius": 0.8,
            "input_scale": 0.1,
        },
        "best_LSTM": {
            "epochs": 400,
            "learning_rate": 0.01,
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
        "best_LSTM": {
            "epochs": 500,
            "learning_rate": 0.02,
            "dropout_rate": 0.0,
            "seed": 42,
            "show_progress": False,
        },
    },
}

# -------------------------  Computation functions  -------------------------
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

# -------------------------  Plotting functions  -------------------------
def LSTM_params(input_dim, hidden_dim, output_dim):
    lstm_layer = 4 * hidden_dim * (hidden_dim + input_dim + 1)
    dense_layer = (hidden_dim + 1) * output_dim
    total = lstm_layer + dense_layer
    return total

def aux_plot_function(df, ax, metric, colors, x_var="num_modes", xlabel="Number of modes/neurons"):
    handles = []

    # --- Quantum models: keep boxplots ---
    for name, sub in df.groupby("label"):
        print("Plotting", name)
        grouped = sub.groupby(x_var)[metric].agg(
            median="median",
            q15=lambda x: x.quantile(0.15),
            q85=lambda x: x.quantile(0.85),
        )
        label = name
        x_vals = np.array(grouped.index)
        med = grouped["median"].values
        q15 = grouped["q15"].values
        q85 = grouped["q85"].values

            # Median line
        ax.scatter(
            x_vals, med,
            color=colors[label],
            lw=2,
            label=label,
            marker="o",
            s=12,
        )

        # Whisker-like error bars
        yerr = np.vstack([med - q15, q85 - med])
        ax.errorbar(
            x_vals, med, yerr=yerr,
            fmt="none",
            ecolor=colors[label],
            elinewidth=1.2,
            capsize=6,
            capthick=1.5,
        )
        
        
        # handles.append(Patch(facecolor=colors[name], edgecolor=colors[name], alpha=0.5, label=name))
        # simplified label
        if "LSTM" in name:
            simple_label = "LSTM"
        elif "ESN" in name:
            simple_label = "ESN"
        else:
            simple_label = name

        if simple_label not in [h.get_label() for h in handles]:
                handles.append(Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=colors[name],
                                markersize=8,
                                alpha=0.6,
                                label=simple_label))

    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=16)

    return handles

def plot_results(df, taskName):
    # Explicit groups
    quantum_label = ["General Encoding"]
    lstm_label = ["best LSTM"]
    fixedNeurons_esn_label = ["best ESN fixedNeurons"]
    fixedParams_esn_label   = ["best ESN fixedParams"]

    # Label sets per plot
    fixedNeurons_labels = lstm_label + fixedNeurons_esn_label + quantum_label
    fixedParams_labels   = fixedParams_esn_label + quantum_label + lstm_label

    # Fixed colors across both panels
    color_map = {
        "General Encoding": "C1",
        "best LSTM": "C0",
        "worst LSTM": "C0",
        "best ESN fixedNeurons": "C3",
        "worst ESN fixedNeurons": "C3",
        "best ESN fixedParams": "C3",   # same as best ESN fixedNeurons
        "worst ESN fixedParams": "C3",  # same as worst ESN fixedNeurons
    }

    metric = "nmse"
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, figure=fig)

    # ---------------------------------------- #
    # -------- Right: fixed Neurons ---------- #
    # ---------------------------------------- #
    subdf_fixedNeurons = df[df["label"].isin(fixedNeurons_labels)]    
    ax = fig.add_subplot(gs[0, 0])

    # Plot and get handles for legend
    handles_fixedNeurons = aux_plot_function(subdf_fixedNeurons, ax, metric, color_map, xlabel="modes/neurons/hidden units")

    # Style adjustments
    ax.set_title("Fixed amount of neurons/modes/hidden units", fontsize=16)
    ax.set_ylabel("NMSE", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xticks([2, 4, 6, 8, 10, 12, 14, 16])
    ax.set_ylim(1e-6,  0.5)
    plt.grid(True, which="both", lw=0.5, ls="--")

    # Highlight Wang et al. 2022
    m1 = df["label"] == "Wang (PhysRevRes.6.043183)"
    m2 = df["dataset"] == taskName
    row = df[m1 & m2].iloc[0]
    ax.scatter(
            row["num_modes"], row["nmse"],
            marker="*", s=80,  # star, large
            color="purple", linewidth=1.2,
            label=row["label"] 
    )

    # Add star to legend
    handles_fixedNeurons.append(Line2D([0], [0], marker="*", color="w",
                                markerfacecolor="purple",
                                markersize=14,
                                label="Wang (PhysRevRes.6.043183)"))

    # Include the legend below both subplots
    fig.legend(handles=handles_fixedNeurons , 
              frameon=True, 
              loc="lower center",
              bbox_to_anchor=(0.5, -0.01),
              ncol=4,
              fontsize=12.5,
    )

    # ---------------------------------------- #
    # ---------- Right: fixedParams ---------- #
    # ---------------------------------------- #
    subdf_fixedParams = df[df["label"].isin(fixedParams_labels)].copy()
    ax = fig.add_subplot(gs[0, 1])

    # Convert modes/"num_modes" to number of parameters for quantum and LSTM models
    # In the LSTM, "num_modes" is the number of hidden units
    quantumMask= subdf_fixedParams["label"] == "General Encoding"
    subdf_fixedParams.loc[quantumMask, "num_modes"] = (
        subdf_fixedParams.loc[quantumMask, "num_modes"] * (subdf_fixedParams.loc[quantumMask, "num_modes"] + 1) // 2
    )
    lstmMask = subdf_fixedParams["label"].str.contains("LSTM")
    subdf_fixedParams.loc[lstmMask, "num_modes"] = LSTM_params(3, subdf_fixedParams.loc[lstmMask, "num_modes"], 3)  # 3 input/output features

    # Plot data. Get handles for legend (not used here)    
    _ = aux_plot_function(subdf_fixedParams, ax, metric, color_map, xlabel="Amount of parameters")
    
    # Highlight Wang et al. 2022
    m1 = df["label"] == "Wang (PhysRevRes.6.043183)"
    m2 = df["dataset"] == taskName
    row = df[m1 & m2].iloc[0]
    
    ax.scatter(
            row["features"], row["nmse"],
            marker="*", s=80,  # star, large
            color="purple", linewidth=1.2,
            label=row["label"] 
    )

     # Style adjustments
    ax.set_ylim(1e-6, 0.5)
    if taskName == "Lorenz":
        ax.set_xlim(6, 110)
    elif taskName == "Double-scroll":
        ax.set_xlim(6, 210)
    ax.set_title("Fixed amount of parameters", fontsize=16)

    plt.grid(True, which="both", lw=0.5, ls="--")
    ax.tick_params(labelsize=14)
    fig.suptitle(f"One-step-ahead prediction ({taskName})", fontsize=20) 
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.27)
    for f in FIGURE_FILES:
        f = f.replace(".png", f"_{taskName}.png").replace(".pdf", f"_{taskName}.pdf")
        print(f"Saving figure to {f}")
        plt.savefig(f)
    return fig

# -------------------------  Analysis functions  -------------------------
def obtain_best_quantum(df):
    # Filter quantum model
    df_quantum = df[df['label'].str.contains("General Encoding")]

    # Get the row with best (lowest) NMSE
    best_row = df_quantum.loc[df_quantum['nmse'].idxmin()]

    # Extract values
    best_nmse = best_row['nmse']
    tau_or_modes = best_row['num_modes']   # or 'tau' if relevant

    # Example: compute error bars as q25/q75 or std
    q25 = df_quantum['nmse'].quantile(0.25)
    q75 = df_quantum['nmse'].quantile(0.75)
    nmse_std = df_quantum['nmse'].std()
    
    subset = df_quantum[df_quantum['num_modes'] == best_row['num_modes']]
    median = subset['nmse'].median()
    q25 = subset['nmse'].quantile(0.25)
    q75 = subset['nmse'].quantile(0.75)
    nmse_std = subset['nmse'].std()
    error_lower = median - q25
    error_upper = q75 - median
    print(f"Best quantum model: NMSE={best_nmse:.6f} (+{error_upper:.6f}/-{error_lower:.6f}) at modes={tau_or_modes}")
    
    return best_nmse, nmse_std , tau_or_modes, 

# -------------------------  Main script  -------------------------
if __name__ == "__main__":
    taskName= sys.argv[1]
    print("Task name:", taskName)
    force_run = False
    num_trials = 25
    pp = ParametricProcess("ktp_780nm_pdc", 180)

    quantum_reservoirs = []
    bestESN_fixedNeurons = [] # Same neurons as quantum modes (not same amount of trainable parameters)
    bestESN_fixedParams = [] # Same neurons as quantum observables (same amount of trainable parameters)
    bestLSTM = []

   # Access optimized parameters
    best_ESN_params = PARAMETERS[taskName]["best_ESN"]
    best_LSTM_params = PARAMETERS[taskName]["best_LSTM"]

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

        bestESN_fixedNeurons.append(
            (
                "best ESN fixedNeurons",
                n, # Same neurons as quantum modes (not same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    n,
                    **best_ESN_params,
                ),
            )
        )

        bestESN_fixedParams.append(
            (
                "best ESN fixedParams",
                equiv_neurons, # Same neurons as quantum observables (same amount of parameters)
                esn.EchoStateNetwork(
                    3,
                    equiv_neurons,
                    **best_ESN_params,
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

       
    path_file = RESULTS_FILE.replace(".csv", f"_{taskName}.csv")
    if force_run or not os.path.exists(path_file):
        print("Computting quantum reservoir results...", len(quantum_reservoirs))
        compute_results(taskName,quantum_reservoirs, num_trials)

        print("Computting best ESN fixedNeurons results...", len(bestESN_fixedNeurons))
        compute_results(taskName, bestESN_fixedNeurons, num_trials)

        print("Computting best ESN fixedParams results...", len(bestESN_fixedParams))
        compute_results(taskName, bestESN_fixedParams, num_trials)

        print("Computting best LSTM results...", len(bestLSTM))
        compute_results(taskName, bestLSTM, num_trials)


    # -----------------   Plot results   ----------------
    df = pd.read_csv(path_file)
    prior_work = pd.DataFrame(
        [
            {
                "label": "Wang (PhysRevRes.6.043183)",
                "trial": 0,
                "nmse": np.mean(np.array([8.9e-3, 11.7e-3, 8.8e-3]) ** 2),
                "features": 14*14,
                "num_modes": 14,
                "dataset": "Double-scroll",
            },
            {
                "label": "Wang (PhysRevRes.6.043183)",
                "trial": 0,
                "nmse": np.mean(np.array([3.7e-3, 14.6e-3, 7.5e-3]) ** 2),
                "features": 8*8,
                "num_modes": 8,
                "dataset": "Lorenz",
            },
        ]
    )
    df = pd.concat([df, prior_work], ignore_index=True)
    plot_results(df, taskName)

    # ------------ Print Best quantum results -----------
    best_quantum_results = obtain_best_quantum(df)
    print("Best quantum results (nmse, nmse_std, tau_or_modes):", best_quantum_results)