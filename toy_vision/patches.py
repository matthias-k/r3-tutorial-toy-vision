"""Patch sampling for natural-image PCA."""
import os

import numpy as np
import matplotlib.pyplot as plt


def _load_gray(path):
    img = plt.imread(path)
    return img[..., :3].mean(-1) if img.ndim == 3 else img


def extract_patches(images, patch=8, n_patches=20000, seed=0):
    """Sample `n_patches` random `patch`x`patch` grayscale patches across `images`.

    `images` is either a directory of image files (png/jpg) or a list of 2-D/3-D
    arrays. Returns an (n_patches, patch*patch) float array. Does NOT download —
    the caller fetches images first and hands them in.
    """
    if isinstance(images, (str, os.PathLike)):
        files = sorted(
            os.path.join(images, f)
            for f in os.listdir(images)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        grays = [_load_gray(f) for f in files]
    else:
        grays = [im[..., :3].mean(-1) if im.ndim == 3 else im for im in images]
    if not grays:
        raise ValueError("extract_patches: no images given")
    rng = np.random.default_rng(seed)
    out = np.empty((n_patches, patch * patch), dtype=float)
    for i in range(n_patches):
        g = grays[rng.integers(len(grays))]
        h, w = g.shape
        y = rng.integers(0, h - patch)
        x = rng.integers(0, w - patch)
        out[i] = g[y:y + patch, x:x + patch].ravel()
    return out
