"""Testing module: how close is a simulated recording to the original?"""
import numpy as np
from spike_detection import remove_drift, estimate_noise, detect_spikes

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


def summarize(t, x):
    """Measure the properties of one recording that a good simulation must reproduce."""
    fs = 1 / np.median(np.diff(t))      # sampling rate from the time array
    flat = remove_drift(x, fs)          # signal with the slow drift removed
    baseline = x - flat                 # the slow drift itself
    spikes = detect_spikes(x, fs)       # sample index of each spike onset
    gaps = np.diff(spikes) / fs         # time between spikes, in seconds
    return {
        "noise_sd": float(estimate_noise(flat)),
        "drift_sd": float(np.std(baseline)),
        "n_spikes": len(spikes),
        "mean_height": float(np.mean(flat[spikes])) if len(spikes) else np.nan,
        "isi_cv": float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 else np.nan,
    }
def compare_files(original_path, simulated_path):
    """Load both files, check structure, measure both, and print them side by side."""
    t_o, x_o = load_recording(original_path)
    t_s, x_s = load_recording(simulated_path)
    structure = check_structure(t_o, t_s)
    orig = summarize(t_o, x_o)
    sim = summarize(t_s, x_s)

    print("Structure:", structure)
    print(f"{'measure':<12} {'original':>9} {'simulated':>10} {'difference':>11}")
    for key in orig:
        diff = (sim[key] - orig[key]) / orig[key] * 100     # percent difference from the original
        print(f"{key:<12} {orig[key]:>9.3f} {sim[key]:>10.3f} {diff:>10.1f}%")
    return structure, orig, sim