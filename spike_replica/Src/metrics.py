"""Accuracy metrics: how close is a generated recording to the real one?

Both recordings go through the SAME fitting pipeline (Src.fit), so every row compares like with like.
Every metric has an `error` (0 = identical) and a `tol` (tolerance). Then
        score  = 100 * max(0, 1 - error/tol)          (0..100)
        passed = score >= 50
The overall score is the plain average of all metric scores. Metrics come in five groups:
drift, spike (shape/rate), timing (spike counts per 0.5 s window), noise, signal (whole-signal statistics).
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy import signal, stats

from .fit import fit
from .generate import generate
from .model import drift, window_counts

PASS_SCORE = 50.0


@dataclass
class Metric:
    """One row of the score table."""
    group: str      # drift / spike / timing / noise / signal
    name: str       # what is measured
    real: float     # value measured on the real data (0 for 'distance' type metrics)
    gen: float      # value measured on the generated data
    error: float    # distance between the two, in the units explained by `tol`
    tol: float      # error at which the score reaches 0

    @property
    def score(self):
        return 100.0 * max(0.0, 1.0 - self.error / self.tol)

    @property
    def passed(self):
        return self.score >= PASS_SCORE


def _rel(a, b):
    """Relative difference |a-b|/|a| (a is the real value)."""
    return abs(a - b) / abs(a)


def _band_logratio(x, y, fs, nperseg, lo, hi, nb=10):
    """Compare two power spectra: average |log10(PSD_x / PSD_y)| over `nb` log-spaced frequency bands.
    0 = identical spectra; 0.3 = a factor of 2 off. Bands average many bins so estimator noise is small."""
    f, px = signal.welch(x, fs, nperseg=nperseg)
    _, py = signal.welch(y, fs, nperseg=nperseg)
    edges = np.geomspace(lo, hi, nb + 1)
    d = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        if m.any():
            d.append(abs(np.log10(px[m].mean() / py[m].mean())))
    return float(np.mean(d))


def _mean_waveform(x, p, onsets, lags=range(-2, 26)):
    """Average drift-free waveform from 2 samples before to 25 samples after isolated spike onsets."""
    t = np.arange(len(x)) / p.fs
    y = x - drift(t, p)                                       # remove drift so only spike + noise remain
    lags = np.array(list(lags))
    keep = [i for k, i in enumerate(onsets)
            if i + lags[0] >= 0 and i + lags[-1] < len(x)                     # window inside the recording
            and np.all(np.abs(np.delete(onsets, k) - i) > lags[-1] + 2)]       # no neighbouring spike inside it
    return np.mean([y[i + lags] for i in keep], axis=0)


def _max_acf_z(r, lags=20):
    """Whiteness statistic: largest |autocorrelation| over lags 1..20, multiplied by sqrt(n) so that
    recordings of different length are comparable (white noise gives about 2-3 whatever n is)."""
    r = r - r.mean()
    d = np.dot(r, r)
    return np.sqrt(len(r)) * max(abs(np.dot(r[:-k], r[k:]) / d) for k in range(1, lags + 1))


def compare(t_real, x_real, t_gen, x_gen, real_fit=None):
    """Compute all metrics. Returns (list[Metric], (params_real, info_real), (params_gen, info_gen)).

    `real_fit` can pass a precomputed fit of the real data (saves time when scoring many seeds).
    The generated recording may have any length.
    """
    pr, ir = real_fit or fit(t_real, x_real)                  # parameters + spikes + residual of the real data
    pg, ig = fit(t_gen, x_gen)                                # same analysis of the generated data
    rr, rg = ir["residual"], ig["residual"]                   # noise = what is left after drift and spikes
    fs = pr.fs
    M = []
    add = lambda g, n, a, b, e, tol: M.append(Metric(g, n, float(a), float(b), float(e), tol))

    # ------------------------------- DRIFT -------------------------------------------------------
    add("drift", "frequency (Hz)", pr.drift_freq, pg.drift_freq, _rel(pr.drift_freq, pg.drift_freq), 0.10)
    add("drift", "drift amplitude", pr.drift_amp, pg.drift_amp, _rel(pr.drift_amp, pg.drift_amp), 0.15)
    add("drift", "offset / amplitude", pr.offset / pr.drift_amp, pg.offset / pg.drift_amp,
        abs(pr.offset - pg.offset) / pr.drift_amp, 0.15)
    # point-by-point difference of the two fitted drift curves over the real recording's time axis
    dr, dg = drift(t_real, pr), drift(t_real, pg)
    e = np.sqrt(np.mean((dr - dg) ** 2)) / pr.drift_amp
    add("drift", "curve RMSE / amp (real t-grid)", 0.0, e, e, 0.25)

    # ------------------------------- SPIKE SHAPE AND RATE ----------------------------------------
    add("spike", "rate (Hz)", pr.spike_rate, pg.spike_rate, _rel(pr.spike_rate, pg.spike_rate), 0.30)
    add("spike", "spike amplitude", pr.spike_amp, pg.spike_amp, _rel(pr.spike_amp, pg.spike_amp), 0.10)
    add("spike", "decay tau (ms)", pr.spike_tau * 1e3, pg.spike_tau * 1e3, _rel(pr.spike_tau, pg.spike_tau), 0.15)
    wr, wg = _mean_waveform(x_real, pr, ir["onsets"]), _mean_waveform(x_gen, pg, ig["onsets"])
    e = np.sqrt(np.mean((wr - wg) ** 2)) / pr.spike_amp       # average spike shape difference
    add("spike", "mean waveform RMSE / amp", 0.0, e, e, 0.15)
    ks = stats.ks_2samp(np.diff(ir["onsets"]), np.diff(ig["onsets"])).statistic   # inter-spike intervals
    add("spike", "inter-spike-interval KS", 0.0, ks, ks, 0.40)

    # ------------------------------- TIMING: SPIKE COUNTS PER 0.5 s WINDOW -------------------------
    w = pr.window_s
    nw_r, nw_g = int(len(x_real) / fs // w), int(len(x_gen) / fs // w)            # number of FULL windows
    cr = window_counts(ir["onsets"], len(x_real), fs, w)[:nw_r]
    cg = window_counts(ig["onsets"], len(x_gen), fs, w)[:nw_g]
    add("timing", "mean count / window", cr.mean(), cg.mean(), _rel(cr.mean(), cg.mean()), 0.30)
    fano = lambda c: c.var() / c.mean() if c.mean() > 0 else 0.0    # Fano factor: 1 for Poisson, >1 bursty
    add("timing", "Fano factor (var/mean)", fano(cr), fano(cg), abs(fano(cr) - fano(cg)), 0.60)
    ks = stats.ks_2samp(cr, cg).statistic                     # same distribution of counts per window?
    add("timing", "count histogram KS", 0.0, ks, ks, 0.35)
    m = min(nw_r, nw_g)                                       # windows both recordings cover (first 10 s)
    mae = np.mean(np.abs(cr[:m] - cg[:m]))                    # window-by-window agreement in time
    add("timing", "window-by-window count MAE", 0.0, mae, mae, 1.0)

    # ------------------------------- NOISE (residual after removing drift and spikes) --------------
    add("noise", "noise std", pr.noise_std, pg.noise_std, _rel(pr.noise_std, pg.noise_std), 0.05)
    ks = stats.ks_2samp(rr, rg).statistic
    add("noise", "distribution KS", 0.0, ks, ks, 0.05)
    sr, sg = (stats.skew(rr), stats.kurtosis(rr)), (stats.skew(rg), stats.kurtosis(rg))
    add("noise", "skew & excess kurtosis (max diff)", max(map(abs, sr)), max(map(abs, sg)),
        max(abs(sr[0] - sg[0]), abs(sr[1] - sg[1])), 0.25)
    ar, ag = _max_acf_z(rr), _max_acf_z(rg)
    add("noise", "whiteness: sqrt(n)*max|acf| lags 1-20", ar, ag, abs(ar - ag), 2.5)
    e = _band_logratio(rr, rg, fs, 128, 8, 500, nb=6)
    add("noise", "PSD mean |log10 ratio| (8-500 Hz)", 0.0, e, e, 0.08)

    # ------------------------------- WHOLE SIGNAL ------------------------------------------------
    e = _band_logratio(x_real, x_gen, fs, 512, 4, 500, nb=8)
    add("signal", "PSD mean |log10 ratio| (4-500 Hz)", 0.0, e, e, 0.15)
    ks = stats.ks_2samp(x_real - drift(t_real, pr), x_gen - drift(t_gen, pg)).statistic
    add("signal", "drift-free amplitude KS", 0.0, ks, ks, 0.10)
    return M, (pr, ir), (pg, ig)


def oracle_r2(x_real, params, onsets):
    """Information only: pointwise R^2 of the NOISE-FREE model (fitted drift + spikes at the real spike
    times) against the real data. Noise is random, so this is the highest R^2 any model can reach."""
    _, x, _ = generate(len(x_real) / params.fs, replace(params, noise_std=0.0), seed=0, onsets=onsets)
    return 1 - np.sum((x_real - x) ** 2) / np.sum((x_real - x_real.mean()) ** 2)


def overall(metrics):
    """Return (overall score, {group: group score}) as plain averages of the metric scores."""
    groups = {}
    for m in metrics:
        groups.setdefault(m.group, []).append(m.score)
    return float(np.mean([m.score for m in metrics])), {g: float(np.mean(v)) for g, v in groups.items()}
