# Wrapper giving a reservoir access to the observables of its most recent
# time steps.

import numpy as np

from .abstract_reservoir import AbstractReservoir


class TemporalObservableConcatenation(AbstractReservoir):
    """
    Reservoir whose observables are the concatenation of the observables
    measured by another reservoir over its most recent time steps.
    """

    def __init__(self, reservoir: AbstractReservoir, num_time_steps: int):
        super().__init__()
        self._reservoir = reservoir
        self._num_time_steps = num_time_steps
        self._ready = False

    @property
    def input_dimension(self) -> int:
        return self._reservoir.input_dimension

    @property
    def output_dimension(self) -> int:
        return self._reservoir.output_dimension * self._num_time_steps

    def reset(self, **kwargs) -> None:
        self._reservoir.reset(**kwargs)
        self._observables = [
            np.zeros(self._reservoir.output_dimension)
        ] * self._num_time_steps
        self._ready = True

    def step(self, s: np.ndarray) -> np.ndarray:
        if not self._ready:
            raise ValueError("reset() must be called before feeding data")

        self._observables = [self._reservoir.step(s)] + self._observables[:-1]
        return np.concatenate(self._observables)
