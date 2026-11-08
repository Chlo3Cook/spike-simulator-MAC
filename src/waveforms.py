import numpy as np


def extract_waveforms(signal, spike_indices, before=2, after=8, search=3):
    """Cut out a window around each spike, aligned on its peak.

    For each spike onset, find the peak within the next `search` samples, then take
    `before` samples before the peak and `after` samples after it. Spikes too close to
    the start or end of the signal for a full window are skipped.

    Returns
    -------
    waveforms : array, shape (n_spikes, before + after + 1)
        One row per spike: its shape around the peak.
    peaks : array of int
        The peak index of each spike kept.
    """
    waveforms = []
    peaks = []
    for i in spike_indices:
        peak = i + np.argmax(signal[i:i + search])   # highest point just after the onset
        start = peak - before
        end = peak + after + 1                       # +1 so the window includes `after` samples past the peak
        if start < 0 or end > len(signal):
            continue                                 # edge case: no full window, skip this spike
        waveforms.append(signal[start:end])
        peaks.append(peak)
    return np.array(waveforms), np.array(peaks, dtype=int)