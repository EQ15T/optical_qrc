# Abstract interface for the simulation of a parametric process.

from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple


class AbstractProcess(ABC):
    """
    Abstract interface for the simulation of a parametric process.

    Subclasses must implement methods for computing (or loading from
    pre-computed values) the phase matching function, computing the pump
    function, and then the Schmidt decomposition of the JSA.
    """

    def __init__(self):
        super().__init__()

    class SVDResult(NamedTuple):
        """
        Container for results of an SVD computation.

        Attributes:
            modes (np.ndarray): Orthogonal supermode functions.
            schmidt_coeffs (np.ndarray): Schmidt coefficients from the SVD.
        """

        modes: np.ndarray
        schmidt_coeffs: np.ndarray

    @abstractmethod
    def compute_svd(self, a_n: np.ndarray, delta_n: np.ndarray) -> SVDResult:
        """
        Compute the SVD of the joint spectral amplitude (JSA).

        Args:
            a_n (np.ndarray): Amplitude of the n frexels.
            delta_n (np.ndarray): Phase shift of the n frexels.

        Returns:
            SVDResult: Orthogonal modes and associated Schmidt coefficients.
        """
        pass
