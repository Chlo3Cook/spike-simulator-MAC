"""Tests for the generator: output format, determinism, signal components and 0.5 s window spike counts."""
import os
from dataclasses import replace

import numpy as np

from Src import Params, generate, load_npz, save_npz
from Src.model import drift, window_counts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = Params.load(os.path.join(ROOT, "params", "fitted_params.json"))      # the fitted parameters
# a version with only spikes (no drift, no noise) so tests can look at exact spike values
SPIKES_ONLY = replace(P, offset=0, drift_amp=0, noise_std=0)


def test_length_and_time_axis_match_original_format():
    # any duration works, and the time axis starts at 0 with 1 ms steps like the original file
    t, x, _ = generate(25.0, P, seed=1)
    assert len(t) == len(x) == 25000 and t[0] == 0 and abs(t[1] - 0.001) < 1e-12 and abs(t[-1] - 24.999) < 1e-9


def test_seed_is_deterministic_and_seeds_differ():
    a, b, c = generate(5, P, seed=3)[1], generate(5, P, seed=3)[1], generate(5, P, seed=4)[1]
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_without_spikes_and_noise_it_is_pure_drift():
    t, x, on = generate(10, replace(P, spike_rate=0, noise_std=0, window_counts=[]), seed=0, spike_mode="poisson")
    assert len(on) == 0 and np.allclose(x, drift(t, P))


def test_spike_shape_is_exponential():
    # a single spike at sample 10: zero before, full amplitude at the onset, then decays by exp(-1/tau) per sample
    _, x, _ = generate(0.1, SPIKES_ONLY, onsets=[10])
    q = SPIKES_ONLY
    assert x[9] == 0 and np.isclose(x[10], q.spike_amp) and np.isclose(x[11], q.spike_amp * np.exp(-1 / (q.spike_tau * q.fs)))


def test_overlapping_spikes_add():
    g = lambda on: generate(0.1, SPIKES_ONLY, onsets=on)[1]
    assert np.allclose(g([10, 14]), g([10]) + g([14]))


def test_window_mode_copies_real_counts_in_first_10s():
    # the spike COUNT in every 0.5 s window of the first 10 s equals the real count
    _, _, on = generate(60, P, seed=9, spike_mode="window")
    c = window_counts(on, 60000, 1000, 0.5)
    assert list(c[:20]) == P.window_counts


def test_window_mode_bootstraps_counts_after_10s():
    # beyond the real recording every window gets a count drawn from the real counts (so never above the real maximum)
    _, _, on = generate(120, P, seed=4, spike_mode="window")
    c = window_counts(on, 120000, 1000, 0.5)
    assert set(c[20:]) <= set(P.window_counts)
    assert abs(c[20:].mean() - np.mean(P.window_counts)) < 0.4           # same average spikes per window


def test_window_mode_positions_are_random_but_inside_window():
    # different seeds put the spikes at different times, but always inside the right window
    o1 = generate(10, P, seed=1)[2]
    o2 = generate(10, P, seed=2)[2]
    assert not np.array_equal(o1, o2)
    assert list(window_counts(o1, 10000, 1000, 0.5)) == list(window_counts(o2, 10000, 1000, 0.5)) == P.window_counts


def test_poisson_mode_does_not_copy_counts():
    # in poisson mode the first-10 s counts are random, not the real ones
    _, _, on = generate(10, P, seed=5, spike_mode="poisson")
    assert list(window_counts(on, 10000, 1000, 0.5)) != P.window_counts


def test_npz_roundtrip_has_original_structure(tmp_path):
    # saved files must have exactly the keys/dtypes of the original recording
    t, x, _ = generate(2, P, seed=0)
    f = tmp_path / "r.npz"
    save_npz(f, t, x)
    d = np.load(f)
    assert set(d.files) == {"t", "raw_data"} and d["t"].dtype == d["raw_data"].dtype == np.float64
    ot, ox = load_npz(os.path.join(ROOT, "data", "spike_data_2.npz"))
    assert d["t"].dtype == ot.dtype and d["raw_data"].ndim == ox.ndim == 1
