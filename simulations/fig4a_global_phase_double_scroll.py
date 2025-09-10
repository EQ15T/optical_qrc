import os

from matplotlib import gridspec
from matplotlib import pyplot as plt

from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.task.benchmarks import doublescroll
from qrc.task.dynamical_system_task import DynamicalSystemTask

FILENAME = os.path.basename(__file__.replace(".py", ""))
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


def plot_results(y_pred, y_true):
    fig = plt.figure(figsize=(6, 3))
    gs = gridspec.GridSpec(3, 1, height_ratios=[0.3, 0.3, 0.3])

    basic_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Subplots for time series
    labels = ["$V_{1}$", "$V_{2}$", "$I$"]
    yrange = [2, 1, 3]

    for i in range(3):
        ax = plt.subplot(gs[i])

        # Plotting time series
        ax.plot(y_true[:, i], "-", linewidth=2, label="Target", color=basic_colors[0])
        ax.plot(
            y_pred[:, i], "--", linewidth=2, label="Predicted", color=basic_colors[1]
        )
        ax.set_ylabel(labels[i], fontsize=14)
        if i == 0:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 0.9),
                ncol=2,
                frameon=False,
                fontsize=12,
            )
            ax.text(1.01, 0.8, "(a)", transform=ax.transAxes, fontsize=13)
        if i == 2:
            ax.set_xlabel("Time", fontsize=12)
            ax.xaxis.set_label_coords(x=0.5, y=-0.08)

        ax.set_ylim(-yrange[i], yrange[i])
        ax.set_yticks([-yrange[i], 0, yrange[i]])
        ax.set_xticks([0, 50, 100, 150])
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.9, bottom=0.13)
    for f in FIGURE_FILES:
        plt.savefig(f)


if __name__ == "__main__":
    L = 5
    noise = 0.05650294519888824

    reservoir = GlobalPhaseProtocol(L * 3, RESERVOIR_PARAMETERS)
    reservoir.reset(noise_level=noise, params_seed=0, noise_seed=0)
    task = DynamicalSystemTask(doublescroll)

    task.run(reservoir, num_washout=3, num_train=350, num_test=150)
    task.train()
    results = task.score()
    plot_results(results.y_pred_test, results.y_true_test)
