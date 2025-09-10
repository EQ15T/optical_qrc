# A task consisting in predicting the dynamics of a system.

from typing import Callable, List, NamedTuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..reservoir.abstract_reservoir import AbstractReservoir
from . import evaluation_metrics
from .abstract_task import AbstractTask


class DynamicalSystemTaskResult(NamedTuple):
    """
    Evaluation results of a regression task.

    Attributes:
        nmse (float): normalized mean square error.
        corrcoeff (float): Pearson correlation coefficient between predicted and true outputs.
        y_pred_test (np.ndarray): Predicted values for the test subset of the sequence.
        y_true_test (np.ndarray): True target values for the test subset of the sequence.
    """

    nmse: float
    corrcoeff: np.ndarray
    y_pred_test: np.ndarray
    y_true_test: np.ndarray


class DynamicalSystemTask(AbstractTask):
    """
    A task consisting in predicting the evolution of a dynamical system, ie
    learning the state of the system at t+dt from the state at t.

    Evaluation can be done in open-loop mode (feeding true states as input)
    or closed-loop mode (feeding the reservoir’s predictions back as input).
    """

    def __init__(
        self, fn: Callable[[np.ndarray], np.ndarray], free_running: bool = False
    ):
        """
        Initialize the task.

        Args:
            fn (Callable[[np.ndarray], np.ndarray]): Function that returns a matrix of
                samples from the dynamical system. Shape should be (num_samples, system_dimension).
            free_running (bool, optional): If True, evaluation uses closed-loop (autoregressive)
                mode. Defaults to False.
        """
        super().__init__()
        self._fn = fn
        self._free_running = free_running
        self._ran = False
        self._trained = False

    def _prepare_input_vector(self, sample: np.ndarray) -> np.ndarray:
        v = np.tile(sample, self._num_repetitions)
        v = v[: self._dimension]
        return v

    def _simulate_reservoir(
        self, r: AbstractReservoir, samples: np.ndarray, steps: int
    ) -> np.ndarray:
        x = np.zeros((steps, r.output_dimension))
        with logging_redirect_tqdm():
            for i in tqdm(range(steps), desc="Reservoir simulation", leave=False):
                input_vector = self._prepare_input_vector(samples[i, :])
                x[i, :] = r.step(input_vector)
        return x

    def _simulate_free_running(self):
        input_vector = self._first_input
        with logging_redirect_tqdm():
            for i in tqdm(
                range(self._num_test),
                desc="Closed-loop reservoir simulation",
                leave=False,
            ):
                observables = self._r.step(self._prepare_input_vector(input_vector))
                self._x[i + self._num_train, :] = observables
                x_scaled = self._scaler.transform(observables.reshape(1, -1))
                input_vector = self._model.predict(x_scaled)[0]

    def run(self, r, num_washout: int, num_train: int, num_test: int) -> None:
        """
        Run the input sequence through the reservoir.

        Args:
            r: Reservoir model instance with `step()` method.
            num_washout (int): Number of washout steps.
            num_train (int): Number of training steps.
            num_test (int): Number of test steps.
            seed (Optional[int]): Random seed for reproducing the same input sequence.
        """
        n = num_washout + num_train + num_test

        s = self._fn(n)
        if s.shape[0] != (n + 1):
            raise ValueError(
                "Wrong number of samples generated from the dynamical system"
            )

        dimension = 1 if len(s.shape) == 1 else s.shape[1]

        # if r.input_dimension % dimension != 0:
        #   raise ValueError('Reservoir input dimension must be a multiple of the dynamical system state space dimension')

        self._num_repetitions = int(np.ceil(r.input_dimension / dimension))
        self._dimension = r.input_dimension

        num_steps = num_washout + num_train if self._free_running else n

        x = self._simulate_reservoir(r, s, num_steps)

        if self._free_running:
            self._x = np.zeros((num_train + num_test, r.output_dimension))
            self._x[:num_train, :] = x[num_washout:, :]
        else:
            self._x = x[num_washout:, :]

        self._y = s[(num_washout + 1) :, :]  # shift for predicting next state
        self._num_train = num_train
        self._num_test = num_test
        self._first_input = s[num_washout + num_train, :]
        self._r = r
        self._ran = True
        self._trained = False

    def train(self, train_subset: Optional[List] = None) -> None:
        """
        Train the readout layer to predict the next state of the dynamical system.

        Args:
            train_subset (Optional[List[int]], optional): Subset of training indices. Defaults to None.

        Raises:
            ValueError: If `run()` has not been called yet.
        """
        if not self._ran:
            raise ValueError("Data not passed through the reservoir, call run()")
        if not train_subset:
            train_subset = (0, self._num_train)
        if train_subset[1] > self._num_train:
            raise ValueError("Incorrect training subset size")

        train_subset = slice(train_subset[0], train_subset[1])
        x_train = self._x[train_subset, :]
        y_train = self._y[train_subset, :]

        scaler = StandardScaler()
        model = LinearRegression()
        x_train = scaler.fit_transform(x_train)
        model.fit(x_train, y_train)

        self._scaler = scaler
        self._model = model
        self._trained = True

    def score(self, plot_results: bool = False) -> DynamicalSystemTaskResult:
        """
        Evaluate the trained model on the test data.

        Args:
            plot_results (bool, optional): If True, plots predictions and observables. Defaults to False.

        Returns:
            DynamicalSystemTask.Result: Named tuple containing evaluation metrics and predictions.

        Raises:
            ValueError: If `train()` has not been called yet.
        """
        if not self._trained:
            raise ValueError("Output layer not trained")

        # If closed-loop evaluation is chosen, the test samples are obtained
        # by letting the reservoir run in closed-loop mode, using the prediction
        # at one step as the input for the next step
        if self._free_running:
            self._simulate_free_running()

        x = self._scaler.transform(self._x)
        y_true = self._y
        y_pred = self._model.predict(x)
        y_pred_test = y_pred[self._num_train :]
        y_true_test = y_true[self._num_train :]

        corrcoeff = evaluation_metrics.correlation_coefficient(y_pred_test, y_true_test)
        nmse = evaluation_metrics.nmse(y_pred_test, y_true_test)

        if plot_results:
            max_num_observables = 4
            x_trimmed = x[self._num_train :, :max_num_observables]

            num_plots = 2 + y_pred.shape[1]

            fig, axs = plt.subplots(num_plots, 1, figsize=(6, num_plots * 3 // 2))
            axs[0].plot(
                x_trimmed, label=[f"observable {i}" for i in range(x_trimmed.shape[1])]
            )
            axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            axs[1].imshow(x[self._num_train :,].T)

            for j in range(y_pred.shape[1]):
                axs[2 + j].plot(y_pred[self._num_train :, j], label="prediction")
                axs[2 + j].plot(y_true[self._num_train :, j], label="true")
                axs[2 + j].set_xlabel("Timestep")
                axs[2 + j].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")

            fig.suptitle(f"NMSE: {nmse:.2f}")
            plt.tight_layout()

        return DynamicalSystemTaskResult(nmse, corrcoeff, y_pred_test, y_true_test)
