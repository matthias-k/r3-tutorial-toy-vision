"""Principal components of image patches."""
import numpy as np


def pca(patches, n_components=None):
    """Principal components of flattened patches, largest variance first.

    `patches` is an (n_patches, patch * patch) array, as `extract_patches` returns.
    Gives back `(components, variance)`: the components as rows, and the fraction
    of the total variance each one explains. `n_components` keeps only the leading
    ones — the fractions still refer to the total, so they no longer sum to 1.
    """
    patches = np.asarray(patches)
    centered = patches - patches.mean(0)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variance = (singular_values ** 2) / (singular_values ** 2).sum()
    if n_components is not None:
        components, variance = components[:n_components], variance[:n_components]
    return components, variance
