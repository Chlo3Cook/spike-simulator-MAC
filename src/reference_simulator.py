"""Reference model (plain Poisson spike times, fitted parameters).
Used only by compare.estimate_spread() to measure how much recordings naturally vary."""
import numpy as np

# All parameters measured from the real recording, in one place
PARAMS = {
    "fs": 1000,               # sampling rate (Hz)
    "noise_sd": 0.30,         # white Gaussian noise (explore_noise.ipynb)
    "drift_offset": 0.0038,   # drift = offset + amp * sin(2*pi*freq*t + phase)  (explore_drift.ipynb)
    "drift_amp": 0.6476,
    "drift_freq": 0.0909,     # Hz
    "drift_phase": 0.2850,    # radians
    "spike_amp": 5.02,        # spike = amp * exp(-t / tau)  (explore_spikes.ipynb)
    "spike_tau": 0.00299,     # seconds (2.99 ms)
    "rate": 4.2,              # Poisson spike rate (Hz)
}


def poisson_spike_times(duration, rate, rng):
    """Random spike times (s) from a Poisson process: gaps between spikes are exponential."""
    times = []
    t = rng.exponential(1 / rate)                 # time of the first spike
    while t < duration:
        times.append(t)
        t += rng.exponential(1 / rate)            # add a random gap to get the next spike
    return np.array(times)


def spike_template(fs, amp, tau, length_s=0.03):
    """One spike: jumps to amp, then decays as exp(-t/tau). Covers 30 ms (10 tau)."""
    k = np.arange(int(length_s * fs)) / fs        # time since the spike, in seconds
    return amp * np.exp(-k / tau)


def simulate(duration, seed=42, **overrides):
    """Simulate a recording: drift + noise + spikes.

    Returns
    -------
    t : time of each sample (s)
    v : simulated voltage (a.u.)
    spike_times : the true spike times (s)
    """
    p = {**PARAMS, **overrides}                   # defaults, with any values you pass replacing them
    rng = np.random.default_rng(seed)
    fs = p["fs"]
    n = int(round(duration * fs))
    t = np.arange(n) / fs

    # Spikes: random times, rounded to the nearest sample, each drawn with the template
    spike_times = poisson_spike_times(duration, p["rate"], rng)
    train = np.zeros(n)
    train[np.round(spike_times * fs).astype(int).clip(0, n - 1)] = 1   # 1 where a spike starts
    spikes = np.convolve(train, spike_template(fs, p["spike_amp"], p["spike_tau"]))[:n]

    # Drift: the sine
    drift = p["drift_offset"] + p["drift_amp"] * np.sin(2 * np.pi * p["drift_freq"] * t + p["drift_phase"])

    # Noise: white Gaussian
    noise = rng.normal(0, p["noise_sd"], n)

    return t, drift + noise + spikes, spike_times