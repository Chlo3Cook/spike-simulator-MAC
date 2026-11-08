"""Testing module: how close is a simulated recording to the original?"""
import numpy as np

REQUIRED_KEYS = ("t", "raw_data")


def load_recording(path):
    """Load an .npz file and return (t, raw_data). Raise an error if the keys or shapes are wrong."""
    data = np.load(path)
    for key in REQUIRED_KEYS:
        if key not in data.files:
            raise ValueError(f"{path} is missing '{key}' (it contains {data.files})")
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["raw_data"], dtype=float)
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x):
        raise ValueError(f"{path}: 't' and 'raw_data' must be 1-D and the same length")
    return t, x


def check_structure(t_orig, t_sim):
    """Return a True/False for each structural property the simulation must share with the original."""
    return {
        "same_length": len(t_orig) == len(t_sim),
        "same_start": bool(np.isclose(t_orig[0], t_sim[0])),
        "same_sampling": bool(np.isclose(np.median(np.diff(t_orig)), np.median(np.diff(t_sim)))),
    }