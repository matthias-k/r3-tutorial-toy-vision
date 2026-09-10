"""toy-vision — tiny shared vision helpers for the r3 tutorial."""
from .patches import extract_patches
from .pca import pca
from .plotting import plot_patches

__all__ = ["extract_patches", "pca", "plot_patches"]
