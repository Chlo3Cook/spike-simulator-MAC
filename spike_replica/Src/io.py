"""Reading and writing .npz files in the same structure as the original recording.

The original file holds two 1-D float64 arrays:
    t         - time stamps in seconds (0, 0.001, 0.002, ...)
    raw_data  - the measured voltage at each time stamp
Every result file this project writes uses exactly the same two keys.
"""
import numpy as np


def load_npz(path):
    """Load a recording and return (t, x) as float arrays. Raises if the structure is not as expected."""
    d = np.load(path)
    t, x = np.asarray(d["t"], float), np.asarray(d["raw_data"], float)
    # sanity check: both must be 1-D and the same length
    if t.ndim != 1 or t.shape != x.shape:
        raise ValueError("expected 1-D arrays 't' and 'raw_data' of equal length")
    return t, x


def save_npz(path, t, x):
    """Save a recording with the same keys and dtype (float64) as the original file."""
    np.savez(path, t=np.asarray(t, np.float64), raw_data=np.asarray(x, np.float64))
