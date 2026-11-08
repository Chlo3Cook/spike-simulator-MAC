import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def detect_onsets(signal, threshold):
    """Return the indices where the signal first rises above the threshold.

    An onset is a sample that is above the threshold when the previous
    sample was not, or when it is the first sample of the signal.
    """
    onsets = []
    for i in range(len(signal)):
        above_now = signal[i] > threshold
        above_before = i > 0 and signal[i - 1] > threshold
        if above_now and not above_before:
            onsets.append(i)
    return np.array(onsets, dtype=int)


def remove_drift(signal, fs, window_s=0.2):
    """Subtract a moving-median baseline to remove slow drift.

    The window must be much longer than a spike (~5 ms) and much shorter
    than the drift (~10 s), so the median follows the drift but ignores spikes.
    """
    window = int(round(window_s * fs)) // 2 * 2 + 1   # seconds → samples, forced odd (201)
    half = window // 2
    padded = np.pad(signal, half, mode="edge")         # repeat edge values so every sample has a full window
    windows = sliding_window_view(padded, window)      # one row per sample: its 201 neighbours
    baseline = np.median(windows, axis=1)              # median of each row
    return signal - baseline


def estimate_noise(signal):
    """Estimate the noise standard deviation with the median absolute deviation (MAD).

    Spikes barely affect the median, so this measures the noise, not the spikes.
    Dividing by 0.6745 converts MAD to standard-deviation units for Gaussian noise.
    """
    return np.median(np.abs(signal - np.median(signal))) / 0.6745


def apply_dead_time(onsets, fs, dead_time_s=0.005):
    """Drop onsets that occur within the dead time of the previous kept onset.

    Prevents a spike's slow tail from being counted as a second spike.
    """
    dead_samples = dead_time_s * fs
    kept = []
    for i in onsets:
        if not kept or i - kept[-1] >= dead_samples:
            kept.append(i)
    return np.array(kept, dtype=int)


def detect_spikes(raw, fs, window_s=0.2, alpha=4.5, dead_time_s=0.008):
    """Full detector: remove drift, estimate noise, threshold, find onsets, apply dead time.

    Returns the sample indices of detected spikes (onset of each above-threshold run).
    """
    flat = remove_drift(raw, fs, window_s)
    sigma = estimate_noise(flat)
    threshold = alpha * sigma
    onsets = detect_onsets(flat, threshold)
    return apply_dead_time(onsets, fs, dead_time_s)

def detect_onsets_numpy(signal, threshold):
    """Vectorized version of detect_onsets: same rule, no Python loop.

    An onset is a sample above the threshold whose previous sample is not above it,
    or the first sample if it is above.
    """
    above = signal > threshold                    # True/False for every sample at once
    onset = np.zeros_like(above)                  # start with all False, same length
    onset[0:1] = above[0:1]                       # first sample: onset if it's above
    onset[1:] = above[1:] & ~above[:-1]           # rest: above now AND NOT above before
    return np.flatnonzero(onset)                  # indices where onset is True