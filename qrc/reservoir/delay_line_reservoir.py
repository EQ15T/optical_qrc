# "Ideal" reservoir with polynomial non-linearity and memory.
# Similar to the NVAR model described in https://www.nature.com/articles/s41467-021-25801-2

import numpy as np
from itertools import combinations_with_replacement

from .abstract_reservoir import AbstractReservoir


class DelayLineReservoir(AbstractReservoir):
    """
    Ideal reservoir with polynomial nonlinearity and memory.

    Builds a feature vector composed of:
    - O_lin: concatenation of delayed inputs
    - O_nonlin: all polynomial features of O_lin, with degree >= 2
    - O_total: concatenation of O_lin and O_nonlin
    """

    def __init__(self, dimension: int, tau: int, max_degree: int = 1):
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
        self._monomials = []
        self._o_lin_dimension = dimension * (tau + 1)
        for o in range(2, max_degree + 1):
            self._monomials.extend(
                list(combinations_with_replacement(range(self._o_lin_dimension), o))
            )

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
        return self._o_lin_dimension + len(self._monomials)

    def reset(self) -> None:
        """
        Reset the internal state (delay line) to zero.

        Returns:
            None
        """
        self._delay_line = np.zeros(self._o_lin_dimension)
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

        self._delay_line = np.concatenate((s, self._delay_line[: -self._dimension]))

        o_lin = self._delay_line
        if self._max_degree >= 2:
            o_nonlin = np.array(
                [np.prod(o_lin[list(idxs)]) for idxs in self._monomials]
            )
        else:
            o_nonlin = np.array([])
        return np.concatenate((o_lin, o_nonlin))
