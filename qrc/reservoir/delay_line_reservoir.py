# "Ideal" reservoir with polynomial non-linearity and memory.

from .abstract_reservoir import AbstractReservoir

import numpy as np


class DelayLineReservoir(AbstractReservoir):
    """
    Ideal reservoir with polynomial nonlinearity and memory.

    At each step, the reservoir computes polynomial features of the current
    input and appends them to a delay line of length `tau`. The output is the
    concatenation of all delayed polynomial features:

        [x(n), x(n)^2, ..., x(n)^k,
         x(n-1), x(n-1)^2, ..., x(n-1)^k,
         ...
         x(n-tau), ..., x(n-tau)^k]
    """

    def __init__(self, dimension: int, tau: int, max_degree: int):
        """
        Initialize the delay-line reservoir.

        Args:
            dimension (int): Input vector dimension.
            tau (int): Memory depth (number of past steps retained).
            max_degree (int): Maximum polynomial degree applied to inputs.
        """
        super().__init__()
        self._dimension = dimension
        self._tau = tau
        self._max_degree = max_degree
        self._ready = False

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
            = (tau + 1) × dimension × max_degree.
        """
        return (self._tau + 1) * self._dimension * self._max_degree

    def reset(self) -> None:
        """
        Reset the internal state (delay line) to zero.

        Returns:
            None
        """
        self._delay_line = np.zeros(self.output_dimension)
        self._ready = True

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

        powers = tuple(s ** (k + 1) for k in range(self._max_degree))
        s_poly = np.concatenate(powers)
        self._delay_line = np.concatenate((self._delay_line[len(s_poly) :], s_poly))
        return self._delay_line
