from .abstract_process import AbstractProcess

import importlib.resources as resources
import numpy as np
import os
import scipy as sp


def open_data_file(file_name: str) -> file:
    """
    Locate and open a binary data file from the local filesystem or package data directory.
    The first match found is opened and returned as a file-like object.

    Args:
        file_name (str): Base name or relative path of the desired file.

    Returns:
        file object: Opened in binary mode, ready for reading.

    Raises:
        FileNotFoundError.
    """
    pkg_data = resources.files(__package__).joinpath("data")
    candidates = [file_name, pkg_data / file_name, pkg_data / (file_name + ".npz")]
    for path in candidates:
        if isinstance(path, str) and os.path.isfile(path):
            return open(path, "rb")
        if not isinstance(path, str) and path.is_file():
            return path.open("rb")
    raise FileNotFoundError(
        f"Could not find '{file_name}' in filesystem or module data/"
    )


class ParametricProcess(object):
    """
    Load and simulate a parametric process with a precomputed phase-matching function.

    This implementation computes the joint spectral amplitude (JSA) from
    stored phase-matching data and a computed pump function, then computes its SVD
    to extract supermodes and Schmidt coefficients.
    """

    def __init__(self, path: str, svd_target_size: Optional[int] = None):
        """
        Initialize the PDC simulation by loading precomputed variables from a file.

        The file must be located in the filesystem or in the package's `data/`
        directory (resolved via `open_data_file`).

        Optionally, the JSA arrays can be truncated symmetrically along both
        axes to reduce the computational cost of the SVD.

        Args:
            path (str): Base filename (with or without `.npz`) of the precomputed data.
            svd_target_size (int, optional): Target size of the JSA before SVD.
                If not set to None, this trims an equal number of rows and
                columns from each side of JSA, reducing the size of the matrix
                decomposed by SVD.

        Raises:
            FileNotFoundError: If the specified file cannot be located.
        """

        variables = np.load(open_data_file(path), allow_pickle=True)
        for name, value in variables.items():
            setattr(self, f"_{name}", value)
        if svd_target_size is not None:
            trim = (self._pump_wavelength.shape[0] - svd_target_size) // 2
            self._pump_wavelength = self._pump_wavelength[trim:-trim, trim:-trim]
            self._phase_matching = self._phase_matching[trim:-trim, trim:-trim]

    def compute_svd(
        self, a_n: np.ndarray, delta_n: np.ndarray
    ) -> AbstractProcess.SVDResult:
        """
        Compute the SVD of the joint spectral amplitude for given pump parameters.

        Args:
            a_n (np.ndarray): Amplitude of the n frexels.
            delta_n (np.ndarray): Phase shift of the n frexels.

        Returns:
            SVDResult: Orthogonal modes and associated Schmidt coefficients.
        """
        pump = self._pump_shape(a_n, delta_n)
        jsa = pump * self._phase_matching

        norm = np.sqrt((np.abs(jsa) ** 2).sum() * self._d_signal * self._d_idler)
        jsa /= norm

        u, s, v = np.linalg.svd(jsa)
        s *= np.sqrt(self._d_signal * self._d_idler)
        u /= np.sqrt(self._d_idler)
        v /= np.sqrt(self._d_signal)
        return AbstractProcess.SVDResult(
            modes=v.T.conj(),
            schmidt_coeffs=s,
        )

    @staticmethod
    def _step(x, middle, width=1e-10):
        """
        Smoothed step function.
        """
        return sp.special.expit(x / width)

    @staticmethod
    def _frexel(a_n, i, pump_wavelength, pump_center, pump_width):
        """
        Create a frequency mask for the i-th mode.
        """
        N = len(a_n)
        offset = pump_center - pump_width
        interval = 2 * pump_width / N  # Dividing pump width into N parts
        return ParametricProcess._step(
            pump_wavelength - (offset + i * interval), 0.5
        ) - ParametricProcess._step(
            pump_wavelength - (offset + (i + 1) * interval), 0.5
        )

    def _pump_shape(self, a_n, delta_n):
        """
        Construct the contribution of the pump to the JSA.

        Args:
            a_n (np.ndarray): Amplitude of the n frexels.
            delta_n (np.ndarray): Phase shift of the n frexels.

        Returns:
            np.ndarray: Complex pump envelope array.
        """
        c = 299792458  # Speed of light in m/s
        N = len(a_n)  # Number of modes in the combination
        res = 0
        for i in range(N):
            phase_shift = np.exp(
                1j * 2 * np.pi * c / self._pump_wavelength * delta_n[i]
            )
            res += (
                a_n[i]
                * phase_shift
                * self._frexel(
                    a_n, i, self._pump_wavelength, self._pump_center, self._pump_width
                )
            )
        return res
