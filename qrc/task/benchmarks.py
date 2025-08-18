# Tasks used as benchmarks.

import numpy as np
from scipy.integrate import solve_ivp


def parity_check(s: np.ndarray, tau: int = 1) -> np.ndarray:
    """
    Compute the parity (mod 2 sum) over a sliding window of size \\tau + 1.

    Args:
        s (np.ndarray): Input binary sequence.
        tau (int): Window size (default 1).

    Returns:
        np.ndarray: Parity-checked sequence of the same length as `s`.
    """
    order = tau + 1
    kernel = np.ones((order,))
    return s if order == 1 else np.fmod(np.convolve(s, kernel)[: -(order - 1)], 2)


def delay(s: np.ndarray, delay: int = 1) -> np.ndarray:
    """
    Shift a sequence by a specified delay, prepending zeros.

    Args:
        s (np.ndarray): Input sequence.
        delay (int): Number of steps to delay (default 1).

    Returns:
        np.ndarray: Delayed sequence of the same length as `s`.
    """
    return s if delay == 0 else np.hstack((np.zeros((delay,)), s[:-delay]))


def narma(s: np.ndarray, tau: int = 2, nu: float = 0.2) -> np.ndarray:
    """
    Generate a NARMA (Nonlinear AutoRegressive Moving Average) sequence.

    Args:
        s (np.ndarray): Input sequence.
        tau (int): Order of the system (default 2).
        nu (float): Scaling factor for the input sequence (default 0.2).

    Returns:
        np.ndarray: Output sequence following the NARMA dynamics.
    """
    # Taken from Eqs. (E1) and (E2) from 10.1103/PhysRevApplied.20.024069
    y = np.zeros_like(s)
    y[0:tau] = s[0:tau]

    s *= nu
    n = len(s)
    if tau > 2:
        for i in range(tau, n):
            y_k = y[i - 1]
            y[i] = (
                0.3 * y_k
                + 0.04 * y_k * np.sum(y[i - tau : i])
                + 1.5 * s[i - 1] * s[i - tau]
                + 0.1
            )
    else:
        for i in range(tau, n):
            y_k = y[i - 1]
            y[i] = 0.4 * y_k + 0.4 * y_k * y[i - 2] + 0.6 * s[i - 1] ** 3 + 0.1
    return y


# Adapted from: https://github.com/quantinfo/ng-rc-paper-code
def doublescroll_step(
    t: float,
    y: list[float],
    r1: float = 1.2,
    r2: float = 3.44,
    r4: float = 0.193,
    alpha: float = 11.6,
    ir: float = 2 * 2.25e-5,
) -> list[float]:
    """
    Compute one time step of the double-scroll chaotic circuit.

    Args:
        t (float): (for ODE solver compatibility).
        y (np.ndarray): State vector [V1, V2, I].
        r1, r2, r4 (float): Circuit resistance R1 (default 1.2).
        alpha (float): Nonlinear gain (default 11.6).
        ir (float): Small current term (default 4.5e-5).

    Returns:
        list[float]: Time derivatives.
    """
    # y[0] = V1, y[1] = V2, y[2] = I
    dV = y[0] - y[1]  # V1-V2
    g = (dV / r2) + ir * np.sinh(alpha * dV)
    dy0 = (y[0] / r1) - g
    dy1 = g - y[2]
    dy2 = y[1] - r4 * y[2]
    return [dy0, dy1, dy2]


def doublescroll(num_points: int, seconds_per_point: float = 1, dt: float = 0.1):
    """
    Simulate the double-scroll chaotic system over a given number of points.

    Args:
        num_points (int): Number of points in the output sequence.
        seconds_per_point (float): Real-time seconds per point (default 1).
        dt (float): Time step for integration (default 0.1).

    Returns:
        np.ndarray: Simulated state sequence with shape (num_points, 3),
                    columns correspond to [V1, V2, I].
    """
    steps_per_point = int(round(seconds_per_point / dt))
    max_time = num_points * seconds_per_point
    integration_steps = num_points * steps_per_point
    t = np.linspace(0, max_time, integration_steps + 1)
    y0 = [0.37926545, 0.058339, -0.08167691]
    solution = solve_ivp(doublescroll_step, (0, max_time), y0, t_eval=t, method="RK23")

    return solution.y[:, ::steps_per_point].T
