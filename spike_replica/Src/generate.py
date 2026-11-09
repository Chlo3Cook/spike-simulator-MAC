"""Generate a synthetic recording of any length from the fitted parameters."""
import numpy as np

from .model import Params, drift, spike_train_signal


def sample_onsets(n, params: Params, rng, spike_mode="window"):
    """Choose the sample index of every spike onset in a recording of n samples.

    spike_mode = "window"  (default): use the real 0.5 s spike counts.
        * The recording is cut into windows of params.window_s seconds.
        * For windows that the real recording also covers (the first 10 s) the spike COUNT per
          window is copied from the real data, so the spike rhythm matches in time. The exact
          position of each spike inside its window is random ("close, not identical").
        * For windows beyond the real recording (e.g. 10-60 s) the count of each window is drawn
          at random (with replacement) from the real counts, so the histogram of counts per
          window stays like the real data while the spike times are new.
    spike_mode = "poisson": no real data used; every sample has probability rate/fs of a spike.
    """
    counts_real = np.asarray(params.window_counts, int)
    if spike_mode == "poisson" or len(counts_real) == 0:    # also the fallback when no counts are stored
        return np.flatnonzero(rng.random(n) < params.spike_rate / params.fs)

    w = int(round(params.window_s * params.fs))             # samples per window
    n_win = -(-n // w)                                      # number of windows needed (ceiling)
    counts = np.empty(n_win, int)
    k = min(len(counts_real), n_win)
    counts[:k] = counts_real[:k]                            # replay real counts where real data exists
    if n_win > k:                                           # bootstrap counts for the extra (new) time
        counts[k:] = rng.choice(counts_real, n_win - k, replace=True)

    onsets = []
    for i, c in enumerate(counts):
        size = min(w, n - i * w)                            # the last window may be shorter
        c = min(int(c), size)                               # cannot place more spikes than samples
        # pick c distinct random samples inside this window, then shift to absolute sample index
        onsets.append(rng.choice(size, c, replace=False) + i * w)
    return np.sort(np.concatenate(onsets)).astype(int)


def generate(duration, params: Params, seed=None, onsets=None, spike_mode="window"):
    """Return (t, raw_data, onsets) for a recording of `duration` seconds.

    duration   - length in seconds (any value, e.g. 10, 60, 600)
    params     - fitted Params (drift, spike shape, noise, window counts)
    seed       - random seed; the same seed always gives the same recording
    onsets     - optional explicit spike sample indices (overrides spike_mode; used in tests / oracle checks)
    spike_mode - "window" or "poisson" (see sample_onsets)
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * params.fs))                    # number of samples
    t = np.arange(n) / params.fs                            # time axis 0, 1/fs, 2/fs, ... like the original
    if onsets is None:
        onsets = sample_onsets(n, params, rng, spike_mode)
    # the three components: slow drift + exponential spikes + white Gaussian noise
    x = drift(t, params) + spike_train_signal(onsets, n, params) + rng.normal(0, params.noise_std, n)
    return t, x, np.asarray(onsets)
