"""End-to-end accuracy tests: how close is generated data to the real recording?"""
import os
from dataclasses import replace

import numpy as np
import pytest

from Src import Params, generate, load_npz
from Src.metrics import compare, overall

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL = load_npz(os.path.join(ROOT, "data", "spike_data_2.npz"))
P = Params.load(os.path.join(ROOT, "params", "fitted_params.json"))


def score(t, x):
    """Overall score and group scores of a generated recording against the real one."""
    return overall(compare(*REAL, t, x)[0])


@pytest.mark.parametrize("duration", [10, 30, 60])
def test_generated_data_is_close_to_real(duration):
    total, groups = score(*generate(duration, P, seed=2024)[:2])
    assert total >= 80, (total, groups)                                   # overall similarity
    assert all(g >= 65 for g in groups.values()), groups                  # no group may be bad


def test_first_10s_spike_timing_matches_real_per_window():
    # the timing group should be near-perfect because counts per 0.5 s window are copied from the real data
    metrics = compare(*REAL, *generate(60, P, seed=1)[:2])[0]
    mae = [m for m in metrics if m.name == "window-by-window count MAE"][0]
    assert mae.error == 0 and mae.passed


def test_saved_results_file_is_close_to_real():
    path = os.path.join(ROOT, "results", "generated_data.npz")
    total, groups = score(*load_npz(path))
    assert total >= 80 and len(load_npz(path)[0]) > 10000, (total, groups)


def test_longer_than_original_recording():
    t, x, _ = generate(60, P, seed=5)
    assert len(x) > len(REAL[1])


def test_metrics_discriminate_wrong_models():
    """The scorer must rank deliberately wrong models clearly below the fitted one."""
    good, _ = score(*generate(30, P, seed=11)[:2])
    wrong = {"no drift": replace(P, drift_amp=0.0),
             "double noise": replace(P, noise_std=2 * P.noise_std),
             "half spikes": replace(P, spike_rate=P.spike_rate / 2, window_counts=[c // 2 for c in P.window_counts]),
             "slow spikes": replace(P, spike_tau=3 * P.spike_tau),
             "wrong drift freq": replace(P, drift_freq=2 * P.drift_freq)}
    for name, q in wrong.items():
        s, _ = score(*generate(30, q, seed=11)[:2])
        assert s < good - 8, (name, s, good)


def test_window_timing_beats_plain_poisson_on_timing_group():
    # using the real 0.5 s counts must give a better "timing" score than ignoring them
    t_w = overall(compare(*REAL, *generate(30, P, seed=3, spike_mode="window")[:2])[0])[1]["timing"]
    t_p = overall(compare(*REAL, *generate(30, P, seed=3, spike_mode="poisson")[:2])[0])[1]["timing"]
    assert t_w > t_p
