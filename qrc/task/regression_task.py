# A task consisting in learning a function of an input sequence.

from ..reservoir.abstract_reservoir import AbstractReservoir
from .abstract_task import AbstractTask
from . import evaluation_metrics

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.autonotebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from typing import List, Optional, Callable, NamedTuple


class RegressionTaskResult(NamedTuple):
    """
    Evaluation results of a regression task.

    Attributes:
        accuracy (float): For binary tasks, classification accuracy; for continuous tasks, correlation coefficient.
        corrcoeff (float): Pearson correlation coefficient between predicted and true outputs.
        y_pred_test (np.ndarray): Predicted values for the test subset of the sequence.
        y_true_test (np.ndarray): True target values for the test subset of the sequence.
    """

    accuracy: float
    corrcoeff: float
    y_pred_test: np.ndarray
    y_true_test: np.ndarray


class RegressionTask(AbstractTask):
    """
    A task consisting in processing a sequence of values, and predicting
    the output of a function computed on this sequence.

    The sequence of inputs used for training are generated randomly.
    """

    def __init__(self, binary: bool = True):
        """
        Initialize the RegressionTask.

        Args:
            binary (bool): Whether to use a binary I/O sequence (default True).
        """
        super().__init__()
        self._ran = False
        self._trained = False
        self._binary = binary

    def run(
        self,
        r: AbstractReservoir,
        num_washout: int,
        num_train: int,
        num_test: int,
        seed: Optional[int] = None,
    ) -> None:
        """
        Run the input sequence through the reservoir.

        Args:
            r: Reservoir model instance with `step()` method.
            num_washout (int): Number of washout steps.
            num_train (int): Number of training steps.
            num_test (int): Number of test steps.
            seed (Optional[int]): Random seed for reproducing the same input sequence.
        """
        if seed is not None:
            np.random.seed(seed)

        n = num_washout + num_train + num_test
        if self._binary:
            s = np.random.randint(0, 2, size=(n,))
        else:
            s = 2 * np.random.rand(n) - 1

        x = np.zeros((n, r.output_dimension))

        with logging_redirect_tqdm():
            for i in tqdm(range(n), desc="Reservoir simulation", leave=False):
                input_vector = np.tile(s[i], r.input_dimension)
                x[i, :] = r.step(np.array(input_vector))

        self._s = s
        self._x = x[num_washout:, :]
        self._num_washout = num_washout
        self._num_train = num_train
        self._ran = True
        self._trained = False

    def train(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        train_subset: Optional[List[int]] = None,
        model_cls=LinearRegression,
    ) -> None:
        """
        Train the readout layer to predict the target function.

        Args:
            fn (Callable[[np.ndarray], np.ndarray]): Target function applied to input sequence.
            train_subset (Optional[List[int]]): Range of training indices (default uses all training data).
            model_cls: Regression model class to use (default LinearRegression).

        Raises:
            ValueError: If `run()` has not been called or training subset is invalid.
        """
        if not self._ran:
            raise ValueError("Data not passed through the reservoir, call run()")

        if not train_subset:
            train_subset = (0, self._num_train)
        if train_subset[1] > self._num_train:
            raise ValueError("Incorrect training subset size")

        train_subset = slice(train_subset[0], train_subset[1])
        x_train = self._x[train_subset, :]
        y_train = fn(self._s)[self._num_washout :][train_subset]

        scaler = StandardScaler()
        model = model_cls()
        x_train = scaler.fit_transform(x_train)
        model.fit(x_train, y_train)

        self._scaler = scaler
        self._model = model
        self._fn = fn
        self._trained = True

    def score(self, plot_results=False) -> RegressionTaskResult:
        """
        Evaluate the trained readout on the full dataset.

        Args:
            plot_results (bool): Whether to plot input, reservoir states, and predictions.

        Returns:
            RegressionTask.Result: Named tuple with accuracy/correlation and predictions.

        Raises:
            ValueError: If the readout has not been trained yet.
        """
        if not self._trained:
            raise ValueError("Output layer not trained")

        x = self._scaler.transform(self._x)
        y_true = self._fn(self._s)[self._num_washout :]
        y_pred = self._model.predict(x)

        y_pred_test = y_pred[self._num_train :]
        y_true_test = y_true[self._num_train :]

        corrcoeff = evaluation_metrics.correlation_coefficient(y_pred_test, y_true_test)

        if self._binary:
            threshold = y_pred_test.mean()
            y_pred_test_thresholded = np.where(y_pred_test >= threshold, 1, 0)
            accuracy = evaluation_metrics.classification_accuracy(
                y_pred_test_thresholded, y_true_test
            )
        else:
            accuracy = corrcoeff

        if plot_results:
            start = self._num_train
            s = self._s[self._num_washout + start :]
            max_num_observables = 4
            x_trimmed = x[start:, :max_num_observables]
            fig, axs = plt.subplots(3, 1, figsize=(10, 6))
            axs[0].plot(s, "o-", label="input")
            axs[0].plot(
                x_trimmed,
                "o-",
                label=[f"observable {i}" for i in range(x_trimmed.shape[1])],
            )
            axs[0].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            axs[1].imshow(x[start:,].T, aspect="auto")
            axs[1].set_yticks([])

            axs[2].plot(y_pred[start:], "o-", label="prediction")
            axs[2].plot(y_true[start:], "s--", label="true")
            axs[2].set_xlabel("Timestep")
            axs[2].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
            axs[2].set_title(f"Accuracy: {(accuracy * 100):.2f}% ; R: {corrcoeff:.2f}")
            plt.tight_layout()

        return RegressionTaskResult(accuracy, corrcoeff, y_pred_test, y_true_test)
