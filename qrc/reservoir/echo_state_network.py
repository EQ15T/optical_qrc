# Implementation of Echo state networks

import numpy as np

from typing import Optional

from .abstract_reservoir import AbstractReservoir


class EchoStateNetwork(AbstractReservoir):
    """
    Echo state network reservoir
    """

    def __init__(
        self,
        dimension: int,
        n: int,
        alpha: float,
        sparsity: float = 0.0,
        spectral_radius: float = 0.8,
        input_scale: float = 1.0,
    ):
        """
        Initialize the reservoir.
        """
        super().__init__()
        self._dimension = dimension
        self._n = n
        self._alpha = alpha
        self._sparsity = sparsity
        self._spectral_radius = spectral_radius
        self._ready = False
        self._input_scale = input_scale

    @property
    def input_dimension(self) -> int:
        """
        Returns:
            int: Dimension of the input vector.
        """
        return self._dimension

    @property
    def output_dimension(self) -> int:
        """
        Returns:
            int: Dimension of the output vector
        """
        return self._n

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the internal state (delay line) to zero.

        Returns:
            None
        """
        rng = np.random.RandomState(seed)
        n = self.output_dimension
        W = rng.randn(n, n)
        mask = rng.rand(n, n) >= self._sparsity
        W *= mask
        eigs = np.linalg.eigvals(W)
        sr = max(abs(eigs)) + 1e-12
        W *= self._spectral_radius / sr
        self._W = W
        self._W_in = rng.randn(n, self.input_dimension)
        self._ready = True
        self._state = np.zeros(n)

    def step(self, s: np.ndarray) -> np.ndarray:
        """
        Perform one update step of the reservoir.

        Args:
            s (np.ndarray): Input vector of shape (dimension,).

        Returns:
            np.ndarray: Updated state of shape (output_dimension,).
        """
        if not self._ready:
            raise ValueError("reset() must be called before feeding data")
        pre = np.tanh(self._W @ self._state + self._W_in @ (s * self._input_scale))
        new_state = (1 - self._alpha) * self._state + self._alpha * np.tanh(pre)
        self._state = new_state
        return new_state
