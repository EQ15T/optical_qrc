# The protocol described in C.1 of arXiv:2506.07279 [quant-ph].

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .abstract_reservoir import AbstractReservoir


@dataclass
class ReservoirParameters:
    """
    Reservoir parameters, fitted from experimental data
    """

    a: float  # Variance scale
    b: float  # Modulation voltage offset
    c: float  # Residual phase shift
    d: float  # Variance offset
    s: float  # Squeezing level
    v_pi_2: float  # Voltage for pi/2 shift


class GlobalPhaseProtocol(AbstractReservoir):
    """
    Reservoir model where a scalar input modulates the pump phase and observables
    (x, p, x+p quadratures) are measured on HG0 + HG1 modes.
    Feedback is applied digitally.
    """

    def __init__(self, num_copies: int, parameters: ReservoirParameters):
        """
        Initialize the reservoir.

        Args:
            num_copies (int): Number of concurrent copies of the system.
            parameters (ReservoirParameters): Physical parameters for variance computation.
        """
        super().__init__()

        self._num_copies = num_copies
        self._reservoir_parameters = parameters
        self._ready = False

    @property
    def input_dimension(self) -> int:
        """
        Returns:
            int: Number of concurrent copies (input dimension).
        """
        return self._num_copies

    @property
    def output_dimension(self) -> int:
        """
        Returns:
            int: Output dimension (3 observables × number of copies).
        """
        # Observables are x, p, x+p on the HG0 + HG1 mode, one for
        # each concurrent copy of the system.
        return self._num_copies * 3

    @staticmethod
    def _sigma_value(
        theta: np.ndarray, noise_level: float, params: ReservoirParameters
    ) -> np.ndarray:
        """
        Compute the quadrature variance for a given total rotation theta, with optional Gaussian noise.

        The rotation includes both the contribution of both the homodyne detection, and the phase shift
        of the pump.

        Args:
            theta (np.ndarray): Homodyne detection angle(s) at which to evaluate the covariance matrix.
            noise_level (float): Standard deviation of additive Gaussian noise.
            params (ReservoirParameters): Reservoir parameters (a, b, c, d, s, v_pi_2).

        Returns:
            np.ndarray: Computed quadrature variance(s), same shape as `theta`.
        """
        theta = np.asarray(theta)
        v = theta / (np.pi / 2) * params.v_pi_2
        theta = np.pi / (2 * (params.v_pi_2 + params.b)) * v + params.c
        noise_dim = len(theta) if np.ndim(theta) > 0 else 1
        o = (
            params.a
            * np.sqrt(
                (params.s**2) * np.cos(theta) ** 2
                + (params.s ** (-2)) * np.sin(theta) ** 2
            )
            + params.d
        )
        o += np.random.normal(loc=0, scale=noise_level, size=noise_dim)
        return o

    def _noisy_observables(
        self, phase: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute noisy covariance matrix elements for the given pump phase.

        Args:
            phase (np.ndarray): Phase angle(s) used to evaluate observables.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Observables corresponding to
            quadrature angles (x, p, x+p)
        """
        measured_quadrature_angle = [np.pi / 2, 0, np.pi / 4]
        return tuple(
            self._sigma_value(phase + q, self._noise_level, self._reservoir_parameters)
            for q in measured_quadrature_angle
        )

    def reset(
        self,
        w: float = 1.25,
        max_squeezing_dB: float = 1.5,
        noise_level: float = 0.0,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = 0,
        fb_mask: Optional[np.ndarray] = None,
        params_seed: Optional[int] = None,
        noise_seed: Optional[int] = None,
    ):
        """
        Reset the reservoir state and initialize feedback mask, noise, and input scaling.

        Args:
            w (float): Base input scaling factor.
            max_squeezing_dB (float): Maximum squeezing (in dB) for feedback normalization.
            noise_level (float): Standard deviation of Gaussian measurement noise.
            alpha (np.ndarray, optional): Input scaling vector of shape (num_copies,).
            beta (np.ndarray, optional): Input offset vector of shape (num_copies,).
            fb_mask (np.ndarray, optional): Feedback matrix of shape (num_copies, num_copies).
            params_seed (int, optional): Random seed for parameter initialization.
            noise_seed (int, optional): Random seed for noise generation.
        """
        N = self._num_copies

        if params_seed is not None:
            np.random.seed(params_seed)

        if alpha is None:
            sign = 2 * np.random.randint(0, 2, size=N) - 1
            magnitude = w + 0.5 * w * (2 * np.random.rand(N) - 1)
            alpha = sign * magnitude

        if fb_mask is None:
            fb_mask = 2 * np.random.rand(N, N) - 1
            max_squeezing = 10 ** (max_squeezing_dB / 10)
            fb_mask = fb_mask / (np.linalg.norm(fb_mask, 2) * max_squeezing)

        if noise_seed is not None:
            # Set seed for future noise calculations
            np.random.seed(noise_seed)

        self._alpha = alpha
        self._beta = beta
        self._fb_mask = fb_mask
        self._noise_level = noise_level
        self._feedback = 0
        self._ready = True

    def step(self, s: np.ndarray) -> np.ndarray:
        """
        Perform one reservoir update step.

        Args:
            s (np.ndarray): Input vector of shape (num_copies,).

        Returns:
            np.ndarray: Concatenated observables (x, p, x+p) of shape (3 × num_copies,).
        """
        if not self._ready:
            raise ValueError("reset() must be called before feeding data")
        observables = self._noisy_observables(
            self._alpha * s + self._feedback + self._beta
        )
        self._feedback = self._fb_mask @ observables[0]
        return np.concatenate(observables)
