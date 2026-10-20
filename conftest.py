"""Shared pytest configuration."""

import numpy as np

# NumPy 2 prints scalars as ``np.float64(2.0)``; keep doctest outputs written
# against the NumPy 1.x repr (``2.0``) valid on both.
if np.lib.NumpyVersion(np.__version__) >= "2.0.0":
    np.set_printoptions(legacy="1.25")
