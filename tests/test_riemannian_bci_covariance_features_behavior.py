"""Behavioral tests for riemannian_bci covariance_features atoms."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.atoms.riemannian_bci.covariance_features.atoms import (
    autocorrelation_covariance_matrix,
    cross_frequency_coherence_matrix,
    riemannian_mean_spd,
    tangent_space_projection,
)


def _make_spd(p: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random SPD matrix of size (p, p)."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = rng.standard_normal((p, p))
    return A @ A.T + p * np.eye(p)


# -- autocorrelation_covariance_matrix ----------------------------------------


def test_autocorrelation_covariance_matrix_shape() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 100))
    n_lags = 5
    result = autocorrelation_covariance_matrix(data, n_lags)
    assert result.shape == (3 * 5, 3 * 5)


def test_autocorrelation_covariance_matrix_symmetric() -> None:
    rng = np.random.default_rng(1)
    data = rng.standard_normal((2, 50))
    result = autocorrelation_covariance_matrix(data, 3)
    np.testing.assert_allclose(result, result.T, atol=1e-12)


def test_autocorrelation_covariance_matrix_positive_semidefinite() -> None:
    rng = np.random.default_rng(2)
    data = rng.standard_normal((4, 200))
    result = autocorrelation_covariance_matrix(data, 4)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals >= -1e-10)


def test_autocorrelation_covariance_matrix_single_lag() -> None:
    rng = np.random.default_rng(3)
    data = rng.standard_normal((2, 30))
    result = autocorrelation_covariance_matrix(data, 1)
    assert result.shape == (2, 2)


# -- cross_frequency_coherence_matrix ----------------------------------------


def test_cross_frequency_coherence_matrix_shape() -> None:
    rng = np.random.default_rng(10)
    data = rng.standard_normal((3, 1000))
    bands = np.array([[1.0, 4.0], [4.0, 8.0], [8.0, 13.0]])
    result = cross_frequency_coherence_matrix(data, fs=256.0, freq_bands=bands, nperseg=128)
    assert result.shape == (3, 3, 3)


def test_cross_frequency_coherence_matrix_diagonal_ones() -> None:
    rng = np.random.default_rng(11)
    data = rng.standard_normal((2, 500))
    bands = np.array([[1.0, 10.0]])
    result = cross_frequency_coherence_matrix(data, fs=100.0, freq_bands=bands, nperseg=64)
    for b in range(bands.shape[0]):
        np.testing.assert_allclose(np.diag(result[b]), 1.0)


def test_cross_frequency_coherence_matrix_symmetric_per_band() -> None:
    rng = np.random.default_rng(12)
    data = rng.standard_normal((4, 2000))
    bands = np.array([[2.0, 6.0], [8.0, 12.0]])
    result = cross_frequency_coherence_matrix(data, fs=256.0, freq_bands=bands, nperseg=128)
    for b in range(bands.shape[0]):
        np.testing.assert_allclose(result[b], result[b].T, atol=1e-12)


def test_cross_frequency_coherence_values_bounded() -> None:
    rng = np.random.default_rng(13)
    data = rng.standard_normal((3, 1000))
    bands = np.array([[1.0, 10.0]])
    result = cross_frequency_coherence_matrix(data, fs=100.0, freq_bands=bands, nperseg=64)
    assert np.all(result >= -1e-10)
    assert np.all(result <= 1.0 + 1e-10)


# -- tangent_space_projection -------------------------------------------------


def test_tangent_space_projection_shape() -> None:
    rng = np.random.default_rng(20)
    p = 3
    n = 5
    matrices = np.array([_make_spd(p, rng) for _ in range(n)])
    reference = _make_spd(p, rng)
    result = tangent_space_projection(matrices, reference)
    assert result.shape == (n, p * (p + 1) // 2)


def test_tangent_space_projection_identity_reference_at_identity() -> None:
    """Projecting the identity at identity reference should give zero vector."""
    p = 3
    identity = np.eye(p)
    matrices = identity[np.newaxis, :, :]
    result = tangent_space_projection(matrices, identity)
    np.testing.assert_allclose(result, 0.0, atol=1e-10)


def test_tangent_space_projection_output_finite() -> None:
    rng = np.random.default_rng(21)
    p = 4
    matrices = np.array([_make_spd(p, rng) for _ in range(3)])
    reference = _make_spd(p, rng)
    result = tangent_space_projection(matrices, reference)
    assert np.isfinite(result).all()


# -- riemannian_mean_spd ------------------------------------------------------


def test_riemannian_mean_of_identical_matrices() -> None:
    """Mean of N copies of the same SPD matrix should be that matrix."""
    rng = np.random.default_rng(30)
    p = 3
    spd = _make_spd(p, rng)
    matrices = np.array([spd for _ in range(5)])
    result = riemannian_mean_spd(matrices)
    np.testing.assert_allclose(result, spd, atol=1e-6)


def test_riemannian_mean_shape() -> None:
    rng = np.random.default_rng(31)
    p = 4
    matrices = np.array([_make_spd(p, rng) for _ in range(6)])
    result = riemannian_mean_spd(matrices)
    assert result.shape == (p, p)


def test_riemannian_mean_symmetric() -> None:
    rng = np.random.default_rng(32)
    p = 3
    matrices = np.array([_make_spd(p, rng) for _ in range(4)])
    result = riemannian_mean_spd(matrices)
    np.testing.assert_allclose(result, result.T, atol=1e-10)


def test_riemannian_mean_single_matrix() -> None:
    """Mean of a single matrix should be that matrix."""
    rng = np.random.default_rng(33)
    p = 3
    spd = _make_spd(p, rng)
    result = riemannian_mean_spd(spd[np.newaxis, :, :])
    np.testing.assert_allclose(result, spd, atol=1e-6)


def test_riemannian_mean_positive_definite() -> None:
    rng = np.random.default_rng(34)
    p = 3
    matrices = np.array([_make_spd(p, rng) for _ in range(5)])
    result = riemannian_mean_spd(matrices)
    eigvals = np.linalg.eigvalsh(result)
    assert np.all(eigvals > 0)
