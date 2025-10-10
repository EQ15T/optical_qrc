import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from tqdm import tqdm

from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.reservoir import long_short_term_memory as lstm
from qrc.task.benchmarks import delay
from qrc.task.regression_task import RegressionTask

from qrc.reservoir.echo_state_network import EchoStateNetwork
from qrc.reservoir.long_short_term_memory import LongShortTermMemory

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
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

ESN_PARAMETERS = {
    "dimension": 1,
    "n": 15,
    "spectral_radius": 0.85,
    "sparsity": 0.1,
    "input_scale": 0.1,
    "alpha": 0.8,
}

LSTM_PARAMETERS = {
    "epochs": 400,
    "learning_rate": 0.01,
    "dropout_rate": 0.0,
    "show_progress": False,
    # "hidden_dim": 1 and 3,  # This will be modified in the loop
    "out_dim": 1,
    "input_dim": 1,
}

def compute_quantum_results(params_dicts: dict, noise_level: float, num_trials: int):
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

            d = {}
            for name in saved_vars:
                d[name] = locals()[name]
            results.append(d)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)

def compute_classical_results(reservoirName, num_trials: int, nRange=None):
    results = []
    saved_vars = ["R", "tau", "capacity", "hidden_neurons"]
    
    R = 1  # Placeholder for R since it's not used in classical reservoirs
    tau_values = range(4)
    for hidden_neurons in nRange if reservoirName == "LSTM" else [ESN_PARAMETERS["n"]]:
        for tau in tqdm(tau_values, desc="Tau values"):
            
            if reservoirName == "ESN":
                reservoir = EchoStateNetwork(**ESN_PARAMETERS)
                
            elif reservoirName == "LSTM":
                reservoir = LongShortTermMemory(hidden_dim=hidden_neurons, **LSTM_PARAMETERS)
                print(reservoir.num_parameters)

                        
            for trial in tqdm(range(num_trials), desc="Trial", leave=False):
                reservoir.reset(seed=trial)

                task = RegressionTask(binary=False)
                task.run(
                    reservoir, num_washout=3, num_train=70, num_test=30, seed=2025 + trial
                )
                task.train(partial(delay, delay=tau))
                capacity = task.score().corrcoeff

                d = {}
                for name in saved_vars:
                    d[name] = locals()[name]
                results.append(d)

    df = pd.DataFrame(results)
    save_path = RESULTS_FILE.replace(".csv", f"_{reservoirName}.csv")
    df.to_csv(save_path, index=False)

def compute_full_stats(df, xcol="training_size", ycol="accuracy"):
    grouped = df.groupby(xcol)[ycol].agg(
        mean="mean",
        median="median",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
        values=lambda x: list(x)  # keep raw values for outlier detection
    ).reset_index()

    
    outliers_dict = {}
    for _, row in grouped.iterrows():
        q25, q75 = row["q25"], row["q75"]
        vals = np.array(row["values"])
        mask = (vals < q25) | (vals > q75)  
        outliers_dict[row[xcol]] = vals[mask] # Everything outside Q25-Q75 is considered an outlier

        print(q25, q75, outliers_dict[row[xcol]])
    print(outliers_dict)
    print("-------")
    return (
        grouped[xcol].values,
        grouped["mean"].values,
        grouped["median"].values,
        grouped["q25"].values,
        grouped["q75"].values,
        outliers_dict,
    )

def plot_grouped_bars_final(datasets, total_group_width=0.8, selected_sizes=None, 
                            xfeature="training_size", yfeature="accuracy", percentage=True,
                            xlabel="Training size", ylabel="XOR accuracy on the test set (%)"):
    all_sizes = sorted(set().union(*[df[xfeature].unique() for df in datasets.values()]))

    if selected_sizes is not None:
        training_sizes = [s for s in sorted(all_sizes) if s in selected_sizes]
    else:
        training_sizes = sorted(all_sizes)


    x = np.arange(len(training_sizes))
    
    bar_width = total_group_width / len(datasets)

    colors = ["C1", "C3", "C0", "cornflowerblue"]  # Manually set colors

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate(datasets.items()): # Iterate over datasets/models
        if percentage:
            df.loc[:, yfeature] = df[yfeature] * 100  # Convert to percentage
        xs, means, medians, q25, q75, outliers_dict = compute_full_stats(df, xcol=xfeature, ycol=yfeature)
        
        # keep only selected sizes
        mask = np.isin(xs, training_sizes)
        xs, means, medians, q25, q75 = xs[mask], means[mask], medians[mask], q25[mask], q75[mask]
        outliers_dict = {k: v for k, v in outliers_dict.items() if k in training_sizes}
        positions = np.array([training_sizes.index(val)  for val in xs])
        
        # Plot bars
        bars = ax.bar(
            x[positions] + i * bar_width,
            medians,
            width=bar_width,
            label=name,
            color=colors[i],
            edgecolor="black",
            alpha=0.8,
        )

   
        # Add quartile ranges as error bars
        yerr = np.vstack([medians - q25, q75 - medians])
        ax.errorbar(
            x[positions] + i * bar_width,
            medians,
            yerr=yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
            capthick=1.5,
        )

        # Plot jittered outliers
        rng = np.random.default_rng(42)  # reproducible jitter
        for xpos, train_size in zip(x[positions] + i * bar_width, xs):
            outs = outliers_dict[train_size]
            if len(outs) > 0:
                jitter = rng.uniform(-bar_width*0.3, bar_width*0.3, size=len(outs))
                ax.scatter(
                    xpos + jitter,
                    outs,
                    color=colors[i],
                    s=25,
                    alpha=0.7,
                    zorder=3,
                    linewidths=0.5, 
                    edgecolors='black'
                )


    ax.set_xticks(x + bar_width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(training_sizes)
    ax.set_xticks(0.3 + np.array(training_sizes), [f"$\\tau={t}$" for t in training_sizes])
    ax.tick_params(labelsize=15)
    ax.legend()
    ax.set_title("Memory task: Performance Comparison", fontsize=20, y=1.15)

    ax.legend(
        bbox_to_anchor=(0.5, 1.15),   # position: 5% right of the axes
        loc="upper center",           # anchor legend's upper-left corner
        borderaxespad=0,
        ncol = 4, 
        frameon=False,
        fontsize=11.5
    )
    
    
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)

    plt.grid(True)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel)
    plt.tight_layout()
    
    for f in FIGURE_FILES:
        plt.savefig(f)

if __name__ == "__main__":
    force_run = False
    plot_results = True

    noise_level = 0.0 # No noise for simulations
    R_values = [5]
    tau_values = range(4)
    params_dicts = [dict(R=r, tau=t) for r in R_values for t in tau_values]
    num_trials = 25

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_quantum_results(params_dicts, noise_level, num_trials)

    file_ESN = RESULTS_FILE.replace(".csv", "_ESN.csv")
    file_LSTM = RESULTS_FILE.replace(".csv", "_LSTM.csv")

    if force_run or not os.path.exists(file_ESN):
        print(f"Computing ESN results and saving to {file_ESN}...")
        compute_classical_results("ESN", num_trials)

    nRange = [1, 3]  # Number of hidden neurons in LSTM for each training size
    if force_run or not os.path.exists(file_LSTM):
        print(f"Computing LSTM results and saving to {file_LSTM}...")
        compute_classical_results("LSTM", num_trials, nRange=nRange)

    # ------------- Plotting ----------------- # 
    if plot_results:
        df_Quantum_STM = pd.read_csv(RESULTS_FILE)
        df_ESN_STM = pd.read_csv(file_ESN)
        df_LSTM_STM = pd.read_csv(file_LSTM)

        df_LSTM_1_STM = df_LSTM_STM[df_LSTM_STM["hidden_neurons"] == 1] # Filter for hidden_neurons = 1
        df_LSTM_3_STM = df_LSTM_STM[df_LSTM_STM["hidden_neurons"] == 3] # Filter for hidden_neurons = 3

        datasets = {"QRC, R=5\n (15 observables - 1 mode)": df_Quantum_STM, "ESN\n (15 neurons)": df_ESN_STM, 
                "LSTM, 400 epochs\n (1 hidden units)": df_LSTM_1_STM, "LSTM, 400 epochs\n (3 hidden units)": df_LSTM_3_STM}

        plot_grouped_bars_final(datasets, selected_sizes=[0, 1, 2, 3],
                        xfeature="tau", yfeature="capacity", percentage=False,
                        xlabel="Delay (tau)", ylabel="Memory capacity (Test set)")

   