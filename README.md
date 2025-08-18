# Optical quantum reservoir computing

This repository contains code for simulating the quantum reservoir computing protocols described in the paper "Experimental memory control in continuous variable optical quantum reservoir computing" by Iris Paparelle, Johan Henaff, Jorge Garcia-Beni, Emilie Gillet, Gian Luca Giorgi, Miguel C. Soriano, Roberta Zambrini and Valentina Parigi [arXiv:2506.07279](https://arxiv.org/abs/2506.07279).

## Setup

```shell
git clone https://github.com/roseopicta/optical_qrc.git
cd optical_qrc
pip install -e .
```

## Reproducing results and figures

Each script in ```simulations``` is intended to be run from the command line and corresponds to a figure (and the underlying simulation) from the manuscript. For example, to reproduce figure 4a:

```bash
python simulations/fig4a_global_phase_double_scroll.py
```

The generated figures are saved in the ```figures``` and raw dataframes in ```results```.

## Code organization

A large part of the code in ```qrc``` can be used, and extended, as a simple library for simulating (quantum) reservoir computing protocols:

* ```parametric_process``` simulates non-linear optical processes. Note that the computation of the phase-matching function is performed with code provided by a third-party. We provide here pre-computed phase-matching data initialized from the parameters of the crystal used in the experiments.
* ```reservoir``` contains simulated models of the two reservoirs described in the paper: ```GlobalPhaseProtocol``` in which the input data (a scalar) is encoded as a phase shift applied to the pump beam, and ```PumpShapingProtocol``` in which the input data (a vector) is encoded in the spectral profile of the pump.
* ```task``` contains classes describing how the reservoir is put to use to solve a specific learning task (data encoding, output layer training, and evaluation). Task and reservoir are decoupled.

The ```simulations``` directory contains the scripts reproducing the figures from the manuscript. Each script can be seen as a specific experiment evaluating an aspect of the simulated system.

## Provided simulations

* ```0_example.py``` is a simple example of how to use the library, and uses the pump shaping protocol to learn a binary function (temporal XOR).
* ```0_general_encoding_double_scroll.py``` compares different parameter sets for the double scroll task, with the general encoding.
* ```fig3c_global_phase_xor.py``` highlights the influence of training size and measurement noise on the XOR task.
* ```fig3e_global_phase_memory.py``` shows how increasing the number of parallel reservoirs improve prediction accuracy on the memory task for an increasing time lag.
* ```fig4a_global_phase_double_scroll.py``` shows the predicted trajectory of the double scroll system (global phase protocol).
* ```fig4b_general_encoding_parity_check.py``` illustrates how different choices of the N and n parameters impact the performance of the pump encoding protocol for a parity check task.
* ```fig4cd_general_encoding_expressivity.py``` studies the correlations between observables for the general encoding scheme.
* ```fig2S_global_phase_double_scroll.py``` shows how the performance of the global phase protocol scales with training set size and number of copies of the reservoir.

## Example use

The following code describe how to use the general encoding (pump shaping) to learn a binary function (temporal XOR).

```python
from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.task.regression_task import RegressionTask
from qrc.task.benchmarks import parity_check

from functools import partial

# Load precomputed variables for PDC simulation
pp = ParametricProcess("ktp_780nm_pdc")

# Use the pump shaping protocol with N=6, n=4.
r = PumpShapingProtocol(6, 4, pp)

# Seed the encoding parameters and reset the reservoir state (feedback)
r.reset(seed=0)

# The reservoir is going to be used to learn a binary function
t = RegressionTask(binary=True)

# Feed the reservoir with a random input sequence, collect output observables
t.run(r, num_washout=10, num_train=180, num_test=50, seed=0)

# Train the output layer to match the target function: a temporal XOR
t.train(partial(parity_check, tau=1))

# Plot the input sequence, observables, and the true and predicted outputs
t.score(plot_results=True)
```

## Dependencies

The code requires the following libraries: numpy (>= 1.22), matplotlib (>= 3.10), pandas, and tdqm
