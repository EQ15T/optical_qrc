# Evaluation metrics.

import numpy as np
import scipy as sp


def classification_accuracy(y_predicted: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the classification accuracy.

    Args:
        y_predicted (np.ndarray): Predicted class labels.
        y_true (np.ndarray): True class labels.

    Returns:
        float: Fraction of correctly predicted labels.
    """
    if y_predicted.shape != y_true.shape:
        raise ValueError("Arrays must have the same size")
    return np.mean(y_true == y_predicted)


def nmse(y_predicted: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the Normalized Mean Squared Error (NMSE).

    Args:
        y_predicted (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

    Returns:
        float: NMSE value.

    Raises:
        ValueError: If `y_predicted` and `y_true` have different shapes.
    """
    if y_predicted.shape != y_true.shape:
        raise ValueError("Arrays must have the same size")
    return np.mean((y_predicted - y_true) ** 2) / np.mean(y_true**2)


def correlation_coefficient(y_predicted: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute the Pearson correlation coefficient from predictions.

    Args:
        y_predicted (np.ndarray): Predicted values.
        y_true (np.ndarray): True values.

    Returns:
        float or np.ndarray: R^2 value for 1D inputs, or array of R^2 values for each variable in 2D inputs.

    Raises:
        ValueError: If `y_predicted` and `y_true` have different shapes.
    """
    if y_predicted.shape != y_true.shape:
        raise ValueError("Arrays must have the same size")

    if len(y_predicted.shape) == 1:
        return np.corrcoef(y_true, y_predicted)[0, 1] ** 2
    else:
        num_variables = y_predicted.shape[1]
        return np.array(
            [
                np.corrcoef(y_true[:, i], y_predicted[:, i])[0, 1] ** 2
                for i in range(num_variables)
            ]
        )


def spectral_distances(y_predicted: np.ndarray, y_true: np.ndarray) -> float:
    if y_predicted.shape != y_true.shape:
        raise ValueError("Arrays must have the same size")

    n, dim = y_predicted.shape

    # Adjust for scale and DC errors
    standardize = lambda x: (x - x.mean()) / x.std()

    magnitude_spectrum_predicted = np.zeros((n // 2 + 1, dim))
    magnitude_spectrum_true = np.zeros((n // 2 + 1, dim))

    total_emd = 0.0
    for i in range(dim):
        # Compute PSD
        f_p, pxx_p = sp.signal.periodogram(
            standardize(y_predicted[:, i]), window="hann"
        )
        f_t, pxx_t = sp.signal.periodogram(standardize(y_true[:, i]), window="hann")
        magnitude_spectrum_predicted[:, i] = pxx_p**0.5
        magnitude_spectrum_true[:, i] = pxx_t**0.5

        # Convert to distribution
        pxx_p /= pxx_p.sum()
        pxx_t /= pxx_t.sum()
        # Compute the EMD between the distributions
        total_emd += sp.stats.wasserstein_distance(
            f_p, f_t, u_weights=pxx_p, v_weights=pxx_t
        )

    total_emd /= dim
    return total_emd, nmse(magnitude_spectrum_predicted, magnitude_spectrum_true)
