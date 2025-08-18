from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.task.benchmarks import doublescroll
from qrc.task.dynamical_system_task import DynamicalSystemTask

from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results", FILENAME + ".pkl")
FIGURE_FILES = [os.path.join(f"figures/{ext}", FILENAME + "." + ext) for ext in ['png', 'pdf']]


RESERVOIR_PARAMETERS = ReservoirParameters(
    a=1.65832624e00,
    b=-2.12645646e-04,
    c=-7.22707129e-02,
    d=-1.19260917e00,
    s=0.8,
    v_pi_2=0.075,
)


def compute_results(params_dicts: dict, num_trials: int, noise: float):
    results = []
    saved_vars = ["L", "M_train", "trial", "corrcoeff"]

    for params in tqdm(params_dicts, desc="Parameter sets"):
        L, M_train = params["L"], params["M_train"]
        reservoir = GlobalPhaseProtocol(L * 3, RESERVOIR_PARAMETERS)
        acc = 0
        for trial in tqdm(range(num_trials), desc="Trial", leave=False):
            reservoir.reset(noise_level=noise, params_seed=trial, noise_seed=trial)
            task = DynamicalSystemTask(doublescroll)
            task.run(
                reservoir, num_washout=3, num_train=M_train, num_test=3 * M_train // 7
            )
            task.train()
            corrcoeff = task.score().corrcoeff
            acc += corrcoeff
            results.append({name: eval(name) for name in saved_vars})

    df = pd.DataFrame(results)
    df.to_pickle(RESULTS_FILE)


def plot_results(df, num_vars=3):
    fig = plt.figure(figsize=(6, 3.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.9, 0.1])
    gs2 = gridspec.GridSpecFromSubplotSpec(1, num_vars, subplot_spec=gs[0], wspace=0.2)

    # Rearrange the data in matrices
    matrices = []
    for i in range(num_vars):
        df["corr_i"] = df["corrcoeff"].apply(lambda x: x[i])
        matrices.append(
            df.pivot_table(
                index="M_train", columns="L", values="corr_i", aggfunc="mean"
            )
        )

    # Shared color normalization
    all_values = np.concatenate(matrices)
    vmin = np.min(all_values * 100)
    vmax = np.max(all_values * 100)
    norm = Normalize(vmin=vmin, vmax=vmax)

    im_list = []
    titles = ["$V_{1}$", "$V_{2}$", "$I$"]

    for i in range(num_vars):
        matrix = matrices[i]
        M_train_vals = matrix.index.to_numpy()
        L_vals = matrix.columns.to_numpy()

        ax = plt.subplot(gs2[i])
        im = ax.pcolormesh(
            M_train_vals,
            L_vals,
            matrix.T * 100,
            cmap="managua",
            shading="auto",
            norm=norm,
        )
        im_list.append(im)

        ax.set_xlabel("Training Size", fontsize=12)
        if i == 0:
            ax.set_ylabel("$L\\ \\  (R = 3×L)$", fontsize=12)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

        # if i == 2:
        #    ax.text(1.15, 0.9, "(b)", transform=ax.transAxes, fontsize=13)
        ax.set_title(titles[i], fontsize=14)

    cbar_ax = plt.subplot(gs[1])
    cbar = plt.colorbar(im_list[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Capacity (%)", fontsize=12)
    plt.tight_layout()
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    force_run = True

    noise = 0.05650294519888824
    num_trials = 5
    params_dicts = [
        dict(L=L, M_train=M_train)
        for L in range(1, 9)
        for M_train in range(70, 560, 70)
    ]

    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(params_dicts, num_trials, noise)

    data = pd.read_pickle(RESULTS_FILE)
    plot_results(data)
