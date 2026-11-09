"""Tests for the fitting code: can it recover known parameters, and what does it find in the real data?"""
import os
from dataclasses import replace

import numpy as np

from Src import Params, fit, generate, load_npz

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# a made-up ground truth, different from the real fit, so we know what the fitter should return
TRUE = Params(offset=0.2, drift_amp=0.8, drift_freq=0.07, drift_phase=1.0, spike_amp=4.0, spike_tau=0.004,
              spike_rate=5.0, noise_std=0.25)


def test_recovers_known_parameters_from_synthetic_10s():
    t, x, on = generate(10, TRUE, seed=7, spike_mode="poisson")
    p, info = fit(t, x)
    assert len(info["onsets"]) == len(on)                                 # every spike found, none invented
    assert abs(p.drift_freq - TRUE.drift_freq) < 0.005 and abs(p.drift_amp - TRUE.drift_amp) < 0.1
    assert abs(p.spike_amp - TRUE.spike_amp) < 0.15 and abs(p.spike_tau - TRUE.spike_tau) < 3e-4
    assert abs(p.noise_std - TRUE.noise_std) < 0.01


def test_detects_overlapping_spikes():
    # spikes only 5 and 2 samples apart (hidden inside each other's tails) must still all be found
    on = [500, 1000, 1005, 1500, 1502, 2000]
    t, x, _ = generate(3, replace(TRUE, spike_rate=0), seed=1, onsets=on)
    assert list(fit(t, x)[1]["onsets"]) == on


def test_window_counts_are_extracted():
    # injected spikes per 0.5 s window must be reproduced exactly by the fit
    t, x, on = generate(10, TRUE, seed=3, spike_mode="poisson")
    p, _ = fit(t, x)
    expected = np.bincount(on // 500, minlength=20)
    assert p.window_counts == expected.tolist() and sum(p.window_counts) == len(on)


def test_real_data_parameters():
    t, x = load_npz(os.path.join(ROOT, "data", "spike_data_2.npz"))
    p, info = fit(t, x)
    assert info["n_spikes"] == 42 and len(p.window_counts) == 20 and sum(p.window_counts) == 42
    assert abs(p.drift_freq - 0.0908) < 0.001 and abs(p.drift_amp - 0.647) < 0.01
    assert abs(p.spike_amp - 5.02) < 0.05 and abs(p.spike_tau - 0.00298) < 1e-4 and abs(p.noise_std - 0.2985) < 0.003
    assert np.isclose(np.std(info["residual"]), p.noise_std, rtol=0.01)
