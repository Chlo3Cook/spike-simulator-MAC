"""Extract drift, spike and noise parameters (and 0.5 s spike counts) from a recording.

Pipeline:
  1. detect_spikes      - rough spike onsets from big upward jumps
  2. grid search        - find the drift frequency (linear least squares at each trial frequency)
  3. joint refinement   - least-squares polish of ALL parameters together
  4. re-detection       - spikes hidden in another spike's tail show up as unexplained jumps in the
                          residual; add them and refit (repeat until nothing new is found)
  5. noise / rate / window counts from the final residual and spike list
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares

from .model import Params, spike_train_signal, window_counts


def detect_spikes(x, fs=1000.0, win_ms=101, k=8.0):
    """Return sample indices of spike onsets.

    1. Remove the slow drift with a running median (101 ms window): r = x - median.
    2. Estimate the noise level robustly with the median absolute deviation (MAD).
    3. A spike onset is a sample where r is above k*sigma AND jumped up by more than k*sigma from the
       previous sample. Requiring the jump stops noise re-crossing the threshold on a decaying tail
       from being counted as a new spike.
    """
    w = int(round(win_ms * fs / 1000)) | 1                   # odd window length in samples
    r = x - median_filter(x, size=w, mode="reflect")
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))     # MAD -> standard deviation
    thr = k * sigma
    step = np.diff(r, prepend=-np.inf)                       # r[i] - r[i-1]  (first sample counts as a jump)
    return np.flatnonzero((r > thr) & (step > thr))


def _design(t, f, spikes):
    """Linear-regression design matrix for a fixed drift frequency f:
    columns = [constant, sin(2 pi f t), cos(2 pi f t), unit-amplitude spike signal]."""
    ang = 2 * np.pi * f * t
    return np.column_stack([np.ones_like(t), np.sin(ang), np.cos(ang), spikes])


def fit(t, x, window_s=0.5):
    """Fit the model to a recording. Returns (Params, info) where info has onsets, residual, std errors."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    fs = round(1.0 / np.median(np.diff(t)), 6)               # sampling rate (rounded to kill float noise)
    n, dur = len(x), len(x) / fs
    onsets = detect_spikes(x, fs)                            # step 1: first spike list

    # ---- step 2: grid search for the drift frequency ---------------------------------------------
    # For a fixed frequency the model is LINEAR in (offset, sin amp, cos amp, spike amp), so ordinary
    # least squares gives the best fit; we keep the frequency with the smallest squared error.
    best = (np.inf, None)
    K = spike_train_signal(onsets, n, Params(fs=fs, spike_amp=1.0, spike_tau=0.003))  # spikes computed once
    for f in np.arange(0.01, 1.0, min(0.002, 0.1 / dur)):    # 0.01..1 Hz; finer step for longer recordings
        X = _design(t, f, K)
        beta, *_ = np.linalg.lstsq(X, x, rcond=None)
        e = np.sum((x - X @ beta) ** 2)
        if e < best[0]:
            best = (e, (beta, f))
    (c, a, b, A), f0 = best[1]

    # ---- steps 3+4: refine everything, then look for spikes the detector merged ------------------
    def resid(q, on):
        """Model minus data for parameters q = [offset, sin amp, cos amp, freq, spike amp, tau]."""
        c, a, b, f, A, tau = q
        ang = 2 * np.pi * f * t
        pp = Params(fs=fs, spike_amp=A, spike_tau=tau)
        return c + a * np.sin(ang) + b * np.cos(ang) + spike_train_signal(on, n, pp) - x

    q = [c, a, b, f0, A, 0.003]
    for _ in range(10):                                      # at most 10 rounds of "fit, then search residual"
        sol = least_squares(resid, q, args=(onsets,), x_scale=[1, 1, 1, 0.01, 1, 0.001])
        q = sol.x
        res = -sol.fun                                       # data minus model
        # a spike that was missed leaves an unexplained jump of ~spike_amp in the residual
        new = np.flatnonzero((res > 0.5 * q[4]) & (np.diff(res, prepend=0) > 0.5 * q[4]))
        new = new[~np.isin(new, onsets)]                     # only genuinely new onsets
        if not len(new):
            break                                            # nothing left unexplained -> done
        onsets = np.union1d(onsets, new)

    # ---- step 5: collect results -------------------------------------------------------------------
    c, a, b, f, A, tau = sol.x
    s2 = np.sum(sol.fun ** 2) / (n - len(sol.x))             # residual variance = noise variance
    # parameter uncertainties from the least-squares Jacobian (standard errors)
    se = np.sqrt(np.diag(np.linalg.pinv(sol.jac.T @ sol.jac)) * s2)
    p = Params(fs=fs, offset=c,
               drift_amp=float(np.hypot(a, b)),              # a*sin + b*cos  ==  hypot(a,b)*sin(. + phase)
               drift_freq=f, drift_phase=float(np.arctan2(b, a)),
               spike_amp=A, spike_tau=tau,
               spike_rate=len(onsets) / dur,                 # maximum-likelihood Poisson rate
               noise_std=float(np.sqrt(s2)),
               window_s=window_s,
               window_counts=[int(v) for v in window_counts(onsets, n, fs, window_s)])  # spikes per 0.5 s
    info = {"onsets": onsets, "residual": -sol.fun,
            "std_errors": dict(zip(["offset", "sin", "cos", "drift_freq", "spike_amp", "spike_tau"], se.tolist())),
            "n_spikes": int(len(onsets))}
    return p, info
