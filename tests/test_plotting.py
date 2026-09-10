"""Tests for toy_vision.plot_patches."""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from toy_vision import plot_patches


def _mosaic(ax):
    """The drawn image with the NaN gaps back as NaN (imshow masks them)."""
    return np.ma.filled(ax.images[0].get_array().astype(float), np.nan)


def test_draws_into_the_axes_it_is_given():
    figure, (left, right) = plt.subplots(1, 2)
    plt.sca(left)          # make the OTHER axes current, so an implementation that
                           # ignored `ax` and drew into plt.gca() would be caught
    returned = plot_patches(np.zeros((4, 9)), ax=right)
    assert returned is right
    # count, don't compare to []: matplotlib's ArtistList doesn't override __eq__, so
    # `left.images == []` is object identity and always False (checked, mpl 3.11.1)
    assert len(right.images) == 1 and len(left.images) == 0
    plt.close(figure)


def test_defaults_to_the_current_axes():
    figure, (left, right) = plt.subplots(1, 2)
    plt.sca(left)
    returned = plot_patches(np.zeros((4, 9)))
    assert returned is left
    assert len(left.images) == 1 and len(right.images) == 0
    plt.close(figure)


def test_default_grid_is_square_ish_and_fits_every_patch():
    figure, ax = plt.subplots()
    plot_patches(np.zeros((5, 9)), ax=ax)          # 5 patches of 3x3 -> 2 rows of 3
    height, width = ax.images[0].get_array().shape
    assert (height, width) == (2 * 4 - 1, 3 * 4 - 1)   # 3 + 1 pad per cell, no trailing pad
    plt.close(figure)


def test_explicit_ncols_wins():
    figure, ax = plt.subplots()
    plot_patches(np.zeros((4, 9)), ax=ax, ncols=4)
    height, width = ax.images[0].get_array().shape
    assert (height, width) == (3, 4 * 4 - 1)
    plt.close(figure)


def test_rejects_patches_that_are_not_square():
    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match="not a square patch"):
        plot_patches(np.zeros((2, 5)), ax=ax)
    plt.close(figure)


def test_tiles_go_in_row_by_row_with_gaps_between():
    patches = np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4])      # three 2x2 patches
    figure, ax = plt.subplots()
    plot_patches(patches, ax=ax, ncols=2, normalize=False)
    mosaic = _mosaic(ax)
    assert mosaic.shape == (5, 5)                              # 2x2 cells of side 2, 1px pad
    assert np.array_equal(mosaic[0:2, 0:2], np.full((2, 2), 1.0))
    assert np.array_equal(mosaic[0:2, 3:5], np.full((2, 2), 2.0))
    assert np.array_equal(mosaic[3:5, 0:2], np.full((2, 2), 3.0))
    assert np.all(np.isnan(mosaic[2, :])) and np.all(np.isnan(mosaic[:, 2]))
    assert np.all(np.isnan(mosaic[3:5, 3:5]))                  # the unused fourth cell
    plt.close(figure)


def test_normalize_scales_each_patch_about_zero():
    patches = np.array([[-1.0, 0.0, 0.0, 1.0], [-100.0, 0.0, 0.0, 100.0]])
    figure, ax = plt.subplots()
    plot_patches(patches, ax=ax, ncols=1)
    assert ax.images[0].get_clim() == (-1.0, 1.0)
    mosaic = _mosaic(ax)
    assert np.array_equal(mosaic[0:2, 0:2], mosaic[3:5, 0:2])  # 100x apart, drawn alike
    plt.close(figure)


def test_without_normalize_the_patches_share_one_scale():
    patches = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
    figure, ax = plt.subplots()
    plot_patches(patches, ax=ax, ncols=1, normalize=False)
    assert ax.images[0].get_clim() == (0.0, 7.0)
    plt.close(figure)


def test_an_all_zero_patch_normalizes_without_dividing_by_zero():
    figure, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plot_patches(np.zeros((2, 4)), ax=ax)
    assert np.all(_mosaic(ax)[0:2, 0:2] == 0.0)
    plt.close(figure)


def test_explicit_nrows_widens_the_grid_instead_of_dropping_patches():
    figure, ax = plt.subplots()
    plot_patches(np.zeros((16, 9)), ax=ax, nrows=2)     # 16 patches of 3x3 -> 2 rows of 8
    height, width = ax.images[0].get_array().shape
    assert (height, width) == (2 * 4 - 1, 8 * 4 - 1)
    plt.close(figure)


def test_normalize_fixes_the_scale_even_when_the_data_does_not_span_it():
    # all-positive patch: normalized it occupies [0.25, 1.0], but the colour scale must
    # still be the symmetric [-1, 1], or two patches couldn't be compared by eye
    figure, ax = plt.subplots()
    plot_patches(np.array([[1.0, 2.0, 3.0, 4.0]]), ax=ax)
    assert ax.images[0].get_clim() == (-1.0, 1.0)
    plt.close(figure)


def test_rejects_input_that_is_not_a_2d_array():
    figure, ax = plt.subplots()
    with pytest.raises(ValueError, match="n_patches, patch"):
        plot_patches(np.zeros((4, 3, 3)), ax=ax)     # unflattened patches
    plt.close(figure)


def test_a_grid_smaller_than_the_patches_shows_the_leading_cells():
    patches = np.array([[1.0] * 4, [2.0] * 4, [3.0] * 4])
    figure, ax = plt.subplots()
    plot_patches(patches, ax=ax, nrows=1, ncols=2, normalize=False)   # room for two of three
    mosaic = _mosaic(ax)
    assert mosaic.shape == (2, 5)
    assert np.array_equal(mosaic[0:2, 0:2], np.full((2, 2), 1.0))
    assert np.array_equal(mosaic[0:2, 3:5], np.full((2, 2), 2.0))
    plt.close(figure)
