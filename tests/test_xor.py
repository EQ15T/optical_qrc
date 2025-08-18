# Not really a unit test :) but serves as an example too.

import pytest

from qrc.parametric_process.precomputed_process import ParametricProcess
from qrc.reservoir.pump_shaping_protocol import PumpShapingProtocol
from qrc.reservoir.global_phase_protocol import GlobalPhaseProtocol, ReservoirParameters
from qrc.task.regression_task import RegressionTask
from qrc.task.benchmarks import parity_check

from functools import partial
import matplotlib.pyplot as plt
import numpy as np


def test_xor_global_phase():
    RESERVOIR_PARAMETERS = ReservoirParameters(
        a=1.61516986e00,
        b=-2.51162304e-04,
        c=-3.37095412e-01,
        d=-1.15273661e00,
        s=0.8,
        v_pi_2=0.075,
    )
    r = GlobalPhaseProtocol(1, RESERVOIR_PARAMETERS)
    t = RegressionTask()
    scale = (np.pi / 2) / RESERVOIR_PARAMETERS.v_pi_2
    r.reset(
        noise_level=0.058,
        alpha=np.pi / 2,
        beta=-0.01 * scale,
        fb_mask=np.array([0.035]) * scale,
        noise_seed=0,
    )
    t.run(r, num_washout=10, num_train=180, num_test=50, seed=0)
    t.train(partial(parity_check, tau=1))
    results = t.score()
    assert results.accuracy >= 0.95, "Accuracy should exceed 95%"


def test_xor_pump_shaping():
    r = PumpShapingProtocol(6, 4, ParametricProcess("ktp_780nm_pdc", 130))
    t = RegressionTask()
    r.reset(seed=0)
    t.run(r, num_washout=10, num_train=180, num_test=50, seed=0)
    t.train(partial(parity_check, tau=1))
    results = t.score()
    assert results.accuracy >= 0.95, "Accuracy should exceed 95%"


if __name__ == "__main__":
    test_xor_global_phase()
    test_xor_pump_shaping()
