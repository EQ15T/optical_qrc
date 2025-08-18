from .abstract_process import AbstractProcess
from .iqo import Process as proc

import numpy as np
from typing import Tuple


DEFAULT_PDC_PARAMS = dict(
    length=15e-3,
    temperature=40,
    grating=np.inf,
    wg_width=3e-6,
    wg_height=3e-6,
    pump_polarization="V",
    signal_polarization="V",
    idler_polarization="V",
    pm_type="sinc",
    process="PDC",
    pump_center=1560e-9 / 2,
    pump_width=2e-9,
    pump_temporal_mode=0,
    pump_type="custom",
    signal_center=1560e-9,
    signal_start=1560e-9 - 200e-9,
    signal_stop=1560e-9 + 200e-9,
    idler_center=1560e-9,
    idler_start=1560e-9 - 200e-9,
    idler_stop=1560e-9 + 200e-9,
    signal_steps=500,
    idler_steps=500,
    crystal="KTP",
)


FAST_PDC_PARAMS = dict(
    length=15e-3,
    temperature=40,
    grating=np.inf,
    wg_width=3e-6,
    wg_height=3e-6,
    pump_polarization="V",
    signal_polarization="V",
    idler_polarization="V",
    pm_type="sinc",
    process="PDC",
    pump_center=1560e-9 / 2,
    pump_width=2e-9,
    pump_temporal_mode=0,
    pump_type="custom",
    signal_center=1560e-9,
    signal_start=1560e-9 - 52e-9,
    signal_stop=1560e-9 + 52e-9,
    idler_center=1560e-9,
    idler_start=1560e-9 - 52e-9,
    idler_stop=1560e-9 + 52e-9,
    signal_steps=130,
    idler_steps=130,
    crystal="KTP",
)


class ParametricProcess(AbstractProcess):
    """
    Concrete implementation of AbstractProcess using the external
    library from IQO

    This class wraps the external library's interface so that it matches the
    AbstractProcess API.
    """

    def __init__(self, parameters: dict):
        self._pp = proc.ParametricProcess()
        self._pp.set_parameters(**parameters)

    def compute_svd(
        self, a_n: np.ndarray, delta_n: np.ndarray
    ) -> AbstractProcess.SVDResult:
        """
        Invoke the external library's SVD computation for the provided pump shape.

        Args:
            a_n (np.ndarray): Amplitude of the n frexels.
            delta_n (np.ndarray): Phase shift of the n frexels.

        Returns:
            SVDResult: Orthogonal modes and associated Schmidt coefficients.
        """
        self._pp.settings["frexel_a_n"] = a_n
        self._pp.settings["frexel_delta_n"] = delta_n
        self._pp.calculate("pump", force_update=True)
        self._pp.calculate("svd")
        results = self._pp.results

        return AbstractProcess.SVDResult(
            modes=results.signal_temporal_modes,
            schmidt_coeffs=results.schmidt_coeffs,
        )
