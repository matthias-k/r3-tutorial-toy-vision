"""Drawing patches — and the components of patches — as an image."""
import numpy as np
import matplotlib.pyplot as plt


def plot_patches(patches, ax=None, nrows=None, ncols=None, pad=1, normalize=True,
                 cmap="gray"):
    """Draw flattened square patches as one tiled image, into `ax`.

    Tiles the patches into a single image rather than making its own grid of
    axes, so this fits as one panel of a larger figure; `ax` defaults to the
    current axes. The grid is square-ish unless you give `nrows`/`ncols`.
    An explicit grid smaller than `patches` shows the leading cells.
    `normalize` scales each patch symmetrically about zero, which is what
    components want — each one has an arbitrary sign and scale. Returns the axes.
    """
    patches = np.asarray(patches)
    if patches.ndim != 2:
        raise ValueError("plot_patches: expected an (n_patches, patch*patch) array")
    n_patches, size = patches.shape
    side = int(round(np.sqrt(size)))
    if side * side != size:
        raise ValueError(f"plot_patches: {size} values per patch is not a square patch")
    if nrows is None and ncols is None:
        ncols = int(np.ceil(np.sqrt(n_patches)))     # square-ish by default
    if ncols is None:
        ncols = int(np.ceil(n_patches / nrows))      # fill the rows we were asked for
    if nrows is None:
        nrows = int(np.ceil(n_patches / ncols))

    tiles = patches[: nrows * ncols].reshape(-1, side, side).astype(float)
    if normalize:
        scale = np.abs(tiles).max(axis=(1, 2), keepdims=True)
        tiles = tiles / np.where(scale == 0, 1, scale)
        vmin, vmax = -1.0, 1.0
    else:
        vmin, vmax = tiles.min(), tiles.max()

    mosaic = np.full((nrows * (side + pad) - pad, ncols * (side + pad) - pad), np.nan)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, ncols)
        top, left = row * (side + pad), column * (side + pad)
        mosaic[top:top + side, left:left + side] = tile

    ax = plt.gca() if ax is None else ax
    ax.imshow(mosaic, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax
