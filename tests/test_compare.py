import numpy as np
import pytest
from compare import load_recording, check_structure, summarize, score, summarize_generated

ORIGINAL = "data/spike_data.npz"


def test_original_matches_itself_perfectly():
    """A file compared with itself must score 0 and pass every structure check."""
    t, x = load_recording(ORIGINAL)
    orig = summarize(t, x)
    distances, overall = score(orig, orig)
    assert overall == 0
    assert all(check_structure(t, t).values())


def test_missing_array_is_rejected(tmp_path):
    """A file without raw_data cannot be compared, so loading it must fail clearly."""
    t, _ = load_recording(ORIGINAL)
    bad_file = tmp_path / "bad.npz"
    np.savez(bad_file, t=t)                      # no raw_data
    with pytest.raises(ValueError):
        load_recording(bad_file)


def test_wrong_length_fails_structure_check():
    """A recording half as long does not have the same structure."""
    t, _ = load_recording(ORIGINAL)
    assert check_structure(t, t[:5000])["same_length"] is False


def test_extra_noise_is_caught():
    """A clearly wrong fake (too much noise) must get a clearly bad score."""
    t, x = load_recording(ORIGINAL)
    noisy = x + np.random.default_rng(0).normal(0, 0.1, len(x))
    distances, overall = score(summarize(t, x), summarize(t, noisy))
    assert distances["noise_sd"] > 3             # noise flagged as clearly off
    assert overall > 2                           # and the overall score shows it


def test_generated_metrics_are_averaged():
    def generate_signal(duration_s, seed):
        rng = np.random.default_rng(seed)
        t = np.arange(int(duration_s * 1000)) / 1000
        return t, rng.normal(0, 1, len(t))

    metrics, generated = summarize_generated(generate_signal, duration=1, n_runs=3)

    expected = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        t = np.arange(1000) / 1000
        expected.append(summarize(t, rng.normal(0, 1, len(t))))
    assert metrics["n_spikes"] == np.mean([row["n_spikes"] for row in expected])
    assert len(generated) == 3
