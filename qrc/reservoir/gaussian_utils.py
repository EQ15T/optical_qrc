# Utilities for manipulating symplectic matrices.

import matplotlib.pyplot as plt
import numpy as np


def complex_to_symplectic(u: np.ndarray) -> np.ndarray:
    """
    Convert a unitary matrix into its real symplectic representation (xxpp convention).

    Args:
        u (np.ndarray): Complex unitary matrix of shape (n, n).

    Returns:
        np.ndarray: Real symplectic matrix of shape (2n, 2n).
    """
    re = np.real(u)
    im = np.imag(u)
    top = np.hstack([re, -im])
    bottom = np.hstack([im, re])
    return np.vstack([top, bottom])


def lambda_to_squeezing(
    lambdas: np.ndarray, alternate_squeezing_xp: bool = True
) -> np.ndarray:
    """
    Build the full covariance matrix (xxpp) from squeezing parameters.

    Even modes have low variance along the p axis, odd modes along the x axis.

    Args:
        lambdas (np.ndarray): Array of squeezing parameters shape (n,).
        alternate_squeezing_xp (bool): If True, alternate orientation of squeezing
            between x (odd indices) and p (even indices).

    Returns:
        np.ndarray: Concatenated array of variances (x variances followed by p variances),
        shape (2n,).
    """
    if alternate_squeezing_xp:
        exponents = np.array([(-1) ** i for i in range(len(lambdas))])
    else:
        exponents = 1
    x_vars = np.exp(2 * lambdas * exponents)
    p_vars = np.exp(-2 * lambdas * exponents)
    return np.concatenate([x_vars, p_vars])


def plot_cov(v: np.ndarray, title: str = None, axs=None):
    """
    Plot a covariance matrix as a heatmap.
    """
    if axs is None:
        plt.figure(figsize=(4, 4))
        axs = plt.gca()
    n, _ = v.shape
    max_v = np.max(v)
    pcm = axs.imshow(v, cmap="RdBu", vmin=-max_v, vmax=max_v)
    plt.colorbar(pcm, ax=axs, fraction=0.046, pad=0.04)
    if title:
        axs.set_title(title)
    axs.set_xticks([])
    axs.set_yticks([])


def unitarity_error(u: np.ndarray) -> float:
    """
    Compute deviation from unitarity for a square matrix.

    Args:
        u (np.ndarray): Candidate unitary matrix of shape (n, n).

    Returns:
        float: Norm of (U^\\dagger U − I), measuring non-unitarity.

    Raises:
        ValueError: If u is not square.
    """
    n, m = u.shape
    if n != m:
        raise ValueError("U must be square")
    return np.linalg.norm(u.conj().T @ u - np.eye(m))
