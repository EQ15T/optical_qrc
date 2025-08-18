# Abstract base class for reservoir implementations.

import numpy as np

from abc import ABC, abstractmethod


class AbstractReservoir(ABC):
    """
    Abstract base class for a reservoir.

    Any specific reservoir implementation should inherit from this class and
    implement the required methods.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def reset(self):
        """
        Reset the reservoir's internal state to its initial conditions.

        This method is typically called before processing a new sequence of
        input data to ensure that the reservoir starts from a clean state.
        """
        pass

    @abstractmethod
    def step(self, s: np.ndarray) -> np.ndarray:
        """
        Perform one update step of the reservoir given an input vector.

        Parameters:
            s (np.ndarray): Input vector at the current time step, of shape (input_dimension,).

        Returns:
            np.ndarray: The updated reservoir state or output, of shape (output_dimension,).

        This method defines the reservoir dynamics (update equation).
        """
        pass

    @property
    @abstractmethod
    def input_dimension(self) -> int:
        """
        The expected dimension of the input vector.

        Returns:
            int: The number of input features that the reservoir accepts at each step.
        """
        pass

    @property
    @abstractmethod
    def output_dimension(self) -> int:
        """
        The dimension of the reservoir's output.

        Returns:
            int: The size of the state vector produced by the reservoir at each step.
        """
        pass
