"""Inter-team evaluator for the reverse-engineering competition.

Usage (terminal):
    python evaluate_team.py data/spike_data.npz team_X_reconstruction.npz

Usage (Python):
    from evaluate_team import evaluate
    result = evaluate("data/spike_data.npz", "team_X_reconstruction.npz")
    print(result["final_score"])

Self-contained: needs only NumPy (Matplotlib optional, for the overlay figure),
so any team can run it without the rest of our repository.

How the score works (0-100, higher is better)
----------------------------------------------
The recording is: drift + spikes + noise. Noise is random and cannot be
reconstructed, so a perfect team's (drift + spikes) leaves ONLY noise behind.
Every metric therefore asks: "how close is what is left over to pure noise?"

  40%  Spike timing  - F1 score: the team's spike_indices vs spikes detected in
                       the original (match within +/-2 ms).
  40%  Fit           - size of what is left over (original minus drift minus
                       spike_signal) compared with the noise floor. 1.0 = exactly
                       noise-sized. Too big = structure missed; too small = noise
                       was copied into the model (overfitting). Both are penalised.
  10%  Leftover structure - what is left over should be white (no pattern);
                       autocorrelation at lags 1-5 beyond chance level (2/sqrt(n)).
  10%  Noise realism - the team's generated noise should have the same SD as
                       the original's noise floor.
An invalid file (missing arrays, wrong length, NaNs) scores 0.
"""
import json
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

REQUIRED = ("t", "drift", "spike_signal", "noise", "reconstruction", "spike_indices")
WEIGHTS = {"spike_f1": 0.40, "fit": 0.40, "whiteness": 0.10, "noise_realism": 0.10}
TOLERANCE_MS = 2          # a team's spike counts as correct if within +/-2 ms of a real one


# ---------- reference spikes: the detector from our Assignment 2B ----------

def _remove_drift(x, fs, window_s=0.2):
    """Subtract a 0.2 s moving median (follows the slow drift, ignores 5 ms spikes)."""
    w = int(round(window_s * fs)) // 2 * 2 + 1
    padded = np.pad(x, w // 2, mode="edge")
    return x - np.median(sliding_window_view(padded, w), axis=1)


def _mad_sigma(x):
    """Robust noise SD (median absolute deviation); barely affected by spikes."""
    return float(np.median(np.abs(x - np.median(x))) / 0.6745)


def detect_spikes(x, fs, alpha=4.5, dead_time_s=0.008):
    """Spike onsets: first sample of each run above alpha * noise SD, with an 8 ms dead time."""
    flat = _remove_drift(x, fs)
    above = flat > alpha * _mad_sigma(flat)
    onsets = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    kept = []
    for i in onsets:
        if not kept or i - kept[-1] >= dead_time_s * fs:
            kept.append(i)
    return np.array(kept, dtype=int)


def noise_floor(x, spikes, fs):
    """Noise SD of the original, independent of anyone's drift model.
    Neighbouring samples share almost the same drift, so x[i+1] - x[i] is mostly
    noise - noise, whose SD is sigma * sqrt(2). The 20 ms after each spike are
    left out so spike tails do not inflate the estimate."""
    keep = np.ones(len(x) - 1, bool)
    for s in spikes:
        keep[max(0, s - 2): s + int(0.020 * fs)] = False
    return _mad_sigma(np.diff(x)[keep]) / np.sqrt(2)


# ---------- metrics ----------

def match_spikes(true_idx, cand_idx, tol):
    """One-to-one matching within +/-tol samples. Returns precision, recall, F1."""
    used = np.zeros(len(true_idx), bool)
    hits = 0
    for c in np.sort(cand_idx):
        d = np.abs(true_idx - c)
        d[used] = tol + 1
        j = int(np.argmin(d)) if len(d) else -1
        if j >= 0 and d[j] <= tol:
            used[j] = True
            hits += 1
    precision = hits / len(cand_idx) if len(cand_idx) else 0.0
    recall = hits / len(true_idx) if len(true_idx) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def autocorr(x, lags=range(1, 6)):
    x = x - x.mean()
    return np.array([np.dot(x[:-k], x[k:]) / np.dot(x, x) for k in lags])


# ---------- validity ----------

def check_validity(ref_t, cand):
    n = len(ref_t)
    v = {"has_all_arrays": all(k in cand for k in REQUIRED)}
    if not v["has_all_arrays"]:
        v["missing"] = [k for k in REQUIRED if k not in cand]
        return v
    v["lengths_ok"] = all(np.shape(cand[k]) == (n,) for k in REQUIRED if k != "spike_indices")
    v["time_matches"] = bool(v["lengths_ok"] and np.allclose(cand["t"], ref_t))
    v["all_finite"] = bool(all(np.all(np.isfinite(cand[k])) for k in REQUIRED))
    idx = np.asarray(cand["spike_indices"])
    v["spike_indices_ok"] = bool(idx.ndim == 1 and np.all(idx == np.round(idx))
                                 and np.all((idx >= 0) & (idx < n)))
    if v["lengths_ok"]:
        structured = cand["drift"] + cand["spike_signal"]
        v["reconstruction_is_drift_plus_spikes"] = bool(np.allclose(cand["reconstruction"], structured, atol=1e-6))
        v["reconstruction_includes_noise"] = bool(np.allclose(cand["reconstruction"], structured + cand["noise"], atol=1e-6))
    v["valid"] = bool(v["lengths_ok"] and v["time_matches"] and v["all_finite"] and v["spike_indices_ok"])
    return v


# ---------- main interface ----------

def evaluate(reference_file, candidate_file, figure_path=None):
    """Score one team's reconstruction against the original recording.

    Returns a dictionary with 'validity', 'metrics', 'final_score' (0-100) and,
    if figure_path is given, the path of a saved overlay figure.
    """
    ref = np.load(reference_file)
    t, x = np.asarray(ref["t"], float), np.asarray(ref["raw_data"], float)
    fs = 1 / np.median(np.diff(t))
    cand = {k: np.asarray(v, float) for k, v in np.load(candidate_file).items()}

    validity = check_validity(t, cand)
    result = {"reference": str(reference_file), "candidate": str(candidate_file), "validity": validity}
    if not validity.get("valid", False):
        result.update(metrics={}, final_score=0.0)
        return result

    # what is left after removing the team's structured signal (works whether or not
    # their 'reconstruction' includes noise, because we use the components directly)
    structured = cand["drift"] + cand["spike_signal"]
    leftover = x - structured
    true_spikes = detect_spikes(x, fs)
    floor = noise_floor(x, true_spikes, fs)
    leftover_sd = float(np.std(leftover))

    precision, recall, f1 = match_spikes(true_spikes, cand["spike_indices"].astype(int),
                                         tol=round(TOLERANCE_MS * fs / 1000))
    ac = autocorr(leftover) if leftover_sd > 1e-12 else np.ones(5)
    noise_sd = float(np.std(cand["noise"]))

    parts = {
        "spike_f1": f1,
        # symmetric: leftover much SMALLER than the noise floor means the team copied
        # noise into their model (overfitting), which is penalised like a poor fit
        "fit": min(floor / max(leftover_sd, 1e-12), leftover_sd / floor),
        # pure noise shows ~2/sqrt(n) autocorrelation by chance; only penalise beyond that
        "whiteness": max(0.0, 1 - max(0.0, float(np.mean(np.abs(ac))) - 2 / np.sqrt(len(x))) / 0.1),
        "noise_realism": max(0.0, 1 - abs(noise_sd / floor - 1) / 0.2),
    }
    final = 100 * sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)

    result["metrics"] = {
        "n_spikes_reference": int(len(true_spikes)),
        "n_spikes_candidate": int(len(cand["spike_indices"])),
        "spike_precision": round(precision, 4),
        "spike_recall": round(recall, 4),
        "noise_floor_sd": round(float(floor), 4),
        "leftover_sd": round(leftover_sd, 4),
        "leftover_autocorr_lags1to5": [round(float(a), 4) for a in ac],
        "candidate_noise_sd": round(noise_sd, 4),
        "part_scores": {k: round(float(v), 4) for k, v in parts.items()},
        "weights": WEIGHTS,
    }
    result["final_score"] = round(float(final), 2)

    if figure_path:
        result["figure"] = _overlay(t, x, structured, cand["spike_indices"].astype(int), true_spikes, figure_path)
    return result


def _overlay(t, x, structured, cand_idx, true_idx, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(14, 6))
    for a, (lo, hi) in zip(ax, [(0, len(t)), (300, 500)]):
        sl = slice(lo, hi)
        a.plot(t[sl], x[sl], lw=0.5, color="0.6", label="original")
        a.plot(t[sl], structured[sl], lw=1, color="tab:red", label="team: drift + spikes")
        a.plot(t[true_idx[(true_idx >= lo) & (true_idx < hi)]], np.full(((true_idx >= lo) & (true_idx < hi)).sum(), x.max() + 0.3), "kv", ms=5, label="reference spikes")
        a.plot(t[cand_idx[(cand_idx >= lo) & (cand_idx < hi)]], np.full(((cand_idx >= lo) & (cand_idx < hi)).sum(), x.max() + 0.8), "rv", ms=5, label="team spikes")
        a.set_xlabel("time (s)")
    ax[0].legend(loc="lower left", fontsize=8, ncol=4)
    ax[0].set_title("Full recording");  ax[1].set_title("Zoom 0.30-0.50 s")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)
    return str(path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python evaluate_team.py REFERENCE.npz CANDIDATE.npz")
    reference, candidate = sys.argv[1], sys.argv[2]
    out_dir = Path("evaluation_results"); out_dir.mkdir(exist_ok=True)
    stem = Path(candidate).stem
    res = evaluate(reference, candidate, figure_path=out_dir / f"{stem}_overlay.png")
    (out_dir / f"{stem}_evaluation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nFINAL SCORE: {res['final_score']}   (saved to {out_dir}/)")
