from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.task.regression_task import RegressionTask
from qrc.task.benchmarks import parity_check

from functools import partial
import matplotlib.pyplot as plt
import numpy as np

# from qrc.parametric_process.process import ParametricProcess, DEFAULT_PDC_PARAMS
# pp = ParametricProcess(DEFAULT_PDC_PARAMS)

pp = ParametricProcess("ktp_780nm_pdc")

# Use the pump shaping protocol with N=6, n=4.
r = PumpShapingProtocol(6, 4, pp)

# Seed the encoding parameters and reset the reservoir state (feedback)
r.reset(seed=0)

# The reservoir is going to be used to learn a binary function
t = RegressionTask()

# Feed the reservoir with a random input sequence, collect output observables
t.run(r, num_washout=10, num_train=180, num_test=50, seed=0)

# Train the output layer to match the target function: a temporal XOR
t.train(partial(parity_check, tau=1))

# Plot the input sequence, observables, and the true and predicted outputs
t.score(plot_results=True)
