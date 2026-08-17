"""Tests for toy_vision.pca."""
import numpy as np

from toy_vision import pca


def _patches(rng, n=500, dim=16):
    return rng.normal(size=(n, dim))


def test_variance_is_descending_fractions_of_the_total():
    components, variance = pca(_patches(np.random.default_rng(0)))
    assert variance.shape == (16,)
    assert np.isclose(variance.sum(), 1.0)
    assert np.all(np.diff(variance) <= 0)


def test_components_are_orthonormal():
    components, _ = pca(_patches(np.random.default_rng(0)))
    assert components.shape == (16, 16)
    assert np.allclose(components @ components.T, np.eye(16), atol=1e-10)


def test_recovers_the_plane_the_data_lives_in():
    # data confined to the plane spanned by dimensions 1 and 3: the leading two
    # components must lie in that plane, and nothing else may explain variance
    rng = np.random.default_rng(1)
    basis = np.zeros((2, 6))
    basis[0, 1] = basis[1, 3] = 1.0
    patches = rng.normal(size=(400, 2)) @ basis
    components, variance = pca(patches)
    assert np.allclose(variance[2:], 0.0, atol=1e-12)
    outside_the_plane = np.delete(components[:2], [1, 3], axis=1)
    assert np.allclose(outside_the_plane, 0.0, atol=1e-8)


def test_n_components_truncates_without_rescaling():
    patches = _patches(np.random.default_rng(0))
    all_components, all_variance = pca(patches)
    components, variance = pca(patches, n_components=4)
    assert components.shape == (4, 16)
    assert np.array_equal(components, all_components[:4])
    assert np.array_equal(variance, all_variance[:4])
    assert variance.sum() < 1.0     # still a fraction of the *total*


def test_subtracts_the_mean_before_decomposing():
    # varies only along dim 1, but sits far from the origin along dim 4. Centered, the
    # leading component is dim 1; uncentered, it would chase the offset instead.
    rng = np.random.default_rng(2)
    patches = np.zeros((300, 6))
    patches[:, 1] = rng.normal(size=300)
    patches[:, 4] = 50.0
    components, variance = pca(patches)
    assert np.isclose(abs(components[0, 1]), 1.0, atol=1e-8)
    assert np.allclose(np.delete(components[0], 1), 0.0, atol=1e-8)
    assert np.isclose(variance[0], 1.0)


def test_variance_fractions_match_the_covariance_spectrum():
    # an independent reference for what "fraction of variance explained" means —
    # catches a spectrum that is descending and sums to 1 but is still wrong
    rng = np.random.default_rng(3)
    patches = rng.normal(size=(600, 5)) * np.array([4.0, 3.0, 2.0, 1.0, 0.5])
    _, variance = pca(patches)
    centered = patches - patches.mean(0)
    eigenvalues = np.sort(np.linalg.eigvalsh(centered.T @ centered))[::-1]
    assert np.allclose(variance, eigenvalues / eigenvalues.sum())
