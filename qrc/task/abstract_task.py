# Abstract base class for a reservoir computing task.

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..reservoir.abstract_reservoir import AbstractReservoir


class AbstractTask(ABC):
    """
    Abstract base class for a reservoir computing task.
    Any specific task (e.g., regression, classification) must inherit from this
    class and implement the required methods.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(
        self, r: AbstractReservoir, num_washout: int, num_train: int, num_test: int
    ) -> None:
        """
        Run the task using the given reservoir model.

        Parameters:
            r: A reservoir model instance that simulates the dynamics of the reservoir.
            num_wash: Number of steps for the washout period.
            num_train: Number of training steps.
            num_test: Number of testing steps.
        """
        pass

    @abstractmethod
    def train(self, train_subset: Optional[List] = None, fn=None) -> None:
        """
        Train the readout layer of the reservoir to match the target function.

        Parameters:
            train_subset: A subset of training data (indices or time steps).
            fn: Optional target function to map inputs to desired outputs.
        """
        pass

    @abstractmethod
    def score(self):
        """
        Compare the predictions with the target function evaluated on inputs.

        Returns:
            score: Task-specific metric (accuracy, RMSE) indicating performance.
        """
        pass
