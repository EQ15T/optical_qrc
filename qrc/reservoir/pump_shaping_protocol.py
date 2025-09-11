# The protocol described in C.2 of arXiv:2506.07279 [quant-ph].

from typing import List, Optional

import numpy as np

from ..parametric_process.abstract_process import AbstractProcess
from . import gaussian_utils
from .abstract_reservoir import AbstractReservoir


def supermode_basis_to_frexel_basis(
    modes: np.ndarray, num_frexels: int, width: int
) -> np.ndarray:
    """
    Compute the change of-basis matrix U by integrating the supermodes over
    frexel bins and normalizing each column.

    Args:
        modes (np.ndarray): A 2D array of mode functions with shape (M, K),
            where M is the number of frequency points and K is the number of modes.
        num_frexels (int): The number of frexel bins (n) to divide the width into.
        width (int): The width of the region being integrated.

    Returns:
        np.ndarray: The complex-valued change-of-basis matrix U with shape (n, n).
            Each column corresponds to an integrated and normalized mode.
    """
    n = num_frexels

    # Precompute slices for bin integration
    start = modes.shape[0] // 2 - width // 2
    band_limit = start + np.floor(np.linspace(0, width, n + 1)).astype(int)

    # Build U from mode integrals
    u = np.array(
        [
            [np.sum(modes[band_limit[i] : band_limit[i + 1], j]) for i in range(n)]
            for j in range(n)
        ]
    )
    # u = np.zeros((n, n), dtype=complex)
    # for i in range(n):
    #     u[:, i] = np.sum(modes[band_limit[i]:band_limit[i + 1], :], axis=0)

    # Normalize each column
    for i in range(n):
        norm = np.linalg.norm(u[:, i])
        if norm > 1e-12:
            u[:, i] /= norm

    # Measure deviation from unitarity is gaussian.unitarity_error(u)
    return u


class PumpShapingProtocol(AbstractReservoir):
    """
    Models a reservoir in which the input data controls the shape of the
    pump beam sent in a non-linear crystal, and in which the measured 'state'
    is the covariance matrix of the state produced by PDC.

    Feedback is implemented digitally.
    """

    def __init__(
        self,
        num_pump_frexels: int,
        num_measured_frexels: int,
        parametric_process: AbstractProcess,
        frexels_width: int = 130,
        use_xp_observables: bool = False,
        alpha_scale: float = 1.0,
        feedback_scale: float = 0.4,
        gain: float = 1.7,
    ):
        """
        Initialize the PumpShapingProtocol.

        Args:
            num_pump_frexels (int): Number of discrete pump frexels (N).
            num_measured_frexels (int): Number of measured frexels (n).
            parametric_process (AbstractProcess):
                The class responsible for computing the SVD of the JSA
                of the parametric process.
            frexels_width (int): Width of the region of spectrum divided into frexels.
            use_xp_observables (bool): Whether to use both x and p observables. The default
                is to use only the x observables (homodyne detection locked to a quadrature).
        """
        super().__init__()

        self._num_pump_frexels = num_pump_frexels  # N
        self._num_measured_frexels = num_measured_frexels  # n
        self._frexels_width = frexels_width
        self._parametric_process = parametric_process
        self._use_xp_observables = use_xp_observables

        # Discretized gaussian profile
        u = np.linspace(-1, 1, num_pump_frexels)
        self._gaussian_profile = np.exp(-(u**2))

        self._gain = gain
        self._alpha_scale = alpha_scale
        self._feedback_scale = feedback_scale

        # Make sure that step() cannot be run before a call to reset()
        self._ready = False

    def _compute_covariance_matrix(
        self, frexel_delta_n: List[float], frexel_a_n: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Compute the covariance matrix for the PDC process.

        This method updates the frexel amplitude and phase parameters,
        recalculates the pump shape and singular value decomposition (SVD)
        of the parametric process, and returns the covariance matrix.

        Args:
            frexel_delta_n (list): frexel phase parameters.
            frexel_a_n (list, optional): Array representing frexel
                amplitude parameters. If None, uses a Gaussian profile.

        Returns:
            np.ndarray: The computed covariance matrix.
        """
        result = self._parametric_process.compute_svd(
            self._gaussian_profile if frexel_a_n is None else frexel_a_n, frexel_delta_n
        )

        n = self._num_measured_frexels
        modes = result.modes[:, :n]
        lambdas = self._gain * result.schmidt_coeffs[0:n]
        d = np.diag(gaussian_utils.lambda_to_squeezing(lambdas))
        u = supermode_basis_to_frexel_basis(modes, n, self._frexels_width)
        s = gaussian_utils.complex_to_symplectic(u)
        sigma = s.T @ d @ s
        return sigma

    @property
    def input_dimension(self) -> int:
        """
        Returns:
            int: Number of pump frexels (input dimension).
        """
        return self._num_pump_frexels

    @property
    def output_dimension(self) -> int:
        """
        Returns:
            int: Number of unique observables (symmetric covariance elements).
        """
        n = self._num_measured_frexels
        if self._use_xp_observables:
            n *= 2
        return n * (n + 1) // 2

    def reset(
        self,
        scale: float = 1e-16,
        alpha: Optional[np.ndarray] = None,
        beta: Optional[np.ndarray] = None,
        fb_mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ):
        """
        Reset the reservoir state, feedback mask, and scaling factors.

        Args:
            scale (float): Scaling factor for feedback computation.
            alpha (np.ndarray, optional): Input scaling vector of shape (num_pump_frexels,).
            beta (np.ndarray, optional): Input offset vector of shape (num_pump_frexels,).
            fb_mask (np.ndarray, optional): Feedback mask of shape (num_pump_frexels, num_measured_frexels).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            None
        """
        N = self._num_pump_frexels
        n = self._num_measured_frexels
        if seed is not None:
            np.random.seed(seed)

        if alpha is None:
            alpha = 2 * np.random.rand(N) - 1
            alpha *= self._alpha_scale

        if beta is None:
            beta = 2 * np.random.rand(N) - 1

        if fb_mask is None:
            fb_mask = 2 * np.random.rand(N, n)
            fb_mask = fb_mask / np.linalg.norm(fb_mask, 2)
            fb_mask *= self._feedback_scale

        self._alpha = alpha
        self._beta = beta
        self._fb_mask = fb_mask
        self._feedback = 0
        self._scale = scale
        self._ready = True

    def step(self, s: np.ndarray) -> np.ndarray:
        """
        Perform one reservoir update step by computing the covariance matrix
        for the modulated pump profile.

        Args:
            s (np.ndarray): Input vector of shape (num_pump_frexels,).

        Returns:
            np.ndarray: Observable vector (upper-triangular covariance elements).
        """
        if not self._ready:
            raise ValueError("reset() must be called before feeding data")

        n = self._num_measured_frexels
        d = self._scale * (self._beta + self._alpha * s + self._feedback)
        sigma = self._compute_covariance_matrix(d.tolist())
        sigma_x = sigma[:n, :n]
        if self._use_xp_observables:
            observables = sigma[np.triu_indices(2 * n)]
        else:
            observables = sigma_x[np.triu_indices(n)]
        self._feedback = np.sum(self._fb_mask * np.diag(sigma_x), axis=1)
        # self._feedback = self._fb_mask @ np.diag(sigma_x)
        return observables
