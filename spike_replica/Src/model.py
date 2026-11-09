"""The signal model and its parameters.

    x(t) = drift(t) + spikes(t) + noise(t)

    drift(t)  = offset + drift_amp * sin(2*pi*drift_freq*t + drift_phase)       (slow sine wave)
    spikes(t) = sum over spike onsets t_k of  spike_amp * exp(-(t - t_k)/spike_tau)   for t >= t_k
                (each spike jumps to spike_amp and decays exponentially; overlapping spikes simply add)
    noise(t)  = independent Gaussian samples, mean 0, standard deviation noise_std (white noise)

On top of these, `window_counts` stores how many spikes the real recording had in every
0.5 s window. The generator uses it so that *when* spikes happen is similar to the real data.
"""
import json
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.signal import lfilter


@dataclass
class Params:
    """All numbers needed to regenerate the signal. The defaults are only placeholders; real values come from fit()."""
    fs: float = 1000.0          # sampling rate in Hz (original: 1000 samples per second)
    offset: float = 0.0         # drift: constant (DC) offset
    drift_amp: float = 0.65     # drift: sine amplitude
    drift_freq: float = 0.09    # drift: sine frequency in Hz (one cycle takes ~11 s)
    drift_phase: float = 0.0    # drift: sine phase in radians
    spike_amp: float = 5.0      # spike: height reached at the onset sample
    spike_tau: float = 0.003    # spike: exponential decay time constant in seconds
    spike_rate: float = 4.0     # spike: average spikes per second (used by the 'poisson' spike mode)
    noise_std: float = 0.3      # noise: standard deviation of the white Gaussian noise
    window_s: float = 0.5       # length of the counting window in seconds
    window_counts: list = field(default_factory=list)   # real spike count in each consecutive window

    def save(self, path, extra=None):
        """Write the parameters to a JSON file (optionally with extra fit information under '_fit_info')."""
        with open(path, "w") as f:
            json.dump({**asdict(self), **({"_fit_info": extra} if extra else {})}, f, indent=2)

    @classmethod
    def load(cls, path):
        """Read parameters back from JSON; keys starting with '_' (extra info) are ignored."""
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")})


def drift(t, p: Params):
    """The slow sine-wave drift evaluated at times t."""
    return p.offset + p.drift_amp * np.sin(2 * np.pi * p.drift_freq * t + p.drift_phase)


def spike_train_signal(onsets, n, p: Params):
    """Voltage contributed by all spikes. `onsets` are sample indices, `n` is the recording length in samples.

    Trick: an exponential decay is a first-order recursive filter y[i] = a*y[i-1] + input[i]
    with a = exp(-1/(tau*fs)). Feeding it a train of impulses (1 at each onset) and scaling by
    spike_amp gives the sum of exponentials, and overlapping spikes add automatically.
    """
    train = np.zeros(n)
    np.add.at(train, np.asarray(onsets, int), 1.0)          # impulse at every onset sample (handles duplicates)
    return p.spike_amp * lfilter([1.0], [1.0, -np.exp(-1.0 / (p.spike_tau * p.fs))], train)


def window_counts(onsets, n_samples, fs, window_s=0.5):
    """Number of spikes in each consecutive window of `window_s` seconds (the last window may be partial).

    Example: 10 s recording, 0.5 s windows -> 20 integers whose sum is the total spike count.
    """
    w = int(round(window_s * fs))                            # samples per window (500 for 0.5 s at 1 kHz)
    n_win = -(-n_samples // w)                               # ceiling division = number of windows
    onsets = np.asarray(onsets, int)
    return np.bincount(onsets // w, minlength=n_win)[:n_win]  # histogram of onset samples into windows
