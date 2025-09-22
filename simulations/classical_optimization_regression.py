
import numpy as np
import os
import pandas as pd
import itertools
import copy
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
from functools import partial

from qrc.task.regression_task import RegressionTask
from qrc.task.benchmarks import delay, parity_check

from qrc.reservoir.echo_state_network import EchoStateNetwork
from qrc.reservoir.long_short_term_memory import LongShortTermMemory

FILENAME = os.path.basename(__file__.replace(".py", ""))
RESULTS_FILE = os.path.join("results/data", FILENAME + ".csv")
FIGURE_FILES = [
    os.path.join(f"results/figures/{ext}", FILENAME + "." + ext)
    for ext in ["png", "pdf"]
]
TABLE_FILE = os.path.join(f"results", FILENAME)
SEED = 42

def compute_single_trial(taskType, label, params, r):
    saved_vars = [
        "taskType",
        "label",
        "accuracy",
        "corrcoeff",
    ]

    fn = partial(delay, delay=3) if taskType == "delay" else partial(parity_check, tau=1)
    binary = True if taskType == "XOR" else False

    t = RegressionTask(binary=binary)
    r.reset(seed=SEED)
    t.run(r, num_washout=5, num_train=90, num_test=30) # Maximum training size for XOR task

    fn = partial(delay, delay=3) if taskType == "delay" else partial(parity_check, tau=1)
    t.train(fn)
    
    accuracy = t.score().accuracy
    corrcoeff = t.score().corrcoeff
    
    d = {}
    for name in saved_vars:
        d[name] = locals()[name]

    d.update(params)

    return d

def compute_results(reservoirs: list, n_jobs=32):
    results = []
    task_types = ["XOR", "delay"]
    for taskType in tqdm(task_types, desc="Task types (XOR/delay)"):
        tasks = []
        for label, params, r in reservoirs:
            tasks.append((taskType, label, params, copy.deepcopy(r)))

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

if __name__ == "__main__":
    force_run = True
   
    params_grid_LSTM = {
        "hidden_dim": [1, 3, 5, 7, 10],  
        "epochs": [100, 200, 300, 400],  
        "learning_rate": [1e-4, 1e-3, 1e-2],  
        "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4],  
        "seed": [SEED],
        "show_progress": [False]
    }

    params_grid_ESN = {
        "n": [3, 9, 15],  
        "alpha": [0.7, 0.8, 0.9, 1.0],  
        "spectral_radius": [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2],  
        "input_scale": [0.1, 0.5, 1.0, 1.5 , 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],  
        "sparsity": [0.0, 0.05, 0.1, 0.15, 0.2], 
    }
  
    reservoirs = []
    for combo in itertools.product(*params_grid_ESN.values()):
        keys = list(params_grid_ESN.keys())
        params = dict(zip(keys, combo))
        reservoirs.append(
            (
                "ESN",
                params,
                EchoStateNetwork(
                    dimension=1, 
                    **{k: v for k, v in params.items()}
                )
            )
        )

    for combo in itertools.product(*params_grid_LSTM.values()):
        keys = list(params_grid_LSTM.keys())
        params = dict(zip(keys, combo))
        reservoirs.append(
            (
                "LSTM",
                params,
                LongShortTermMemory(
                    input_dim=1, 
                    out_dim=1,
                    **{k: v for k, v in params.items()}
                )
            )
        )

    filename = os.path.basename(__file__.replace(".py", ""))
   
    if force_run or not os.path.exists(RESULTS_FILE):
        compute_results(reservoirs)
