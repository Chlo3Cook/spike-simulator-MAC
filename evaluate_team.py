"""Table-formatted evaluator for Task 3 team reconstructions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

REQUIRED = ("t", "drift", "spike_signal", "noise", "reconstruction", "spike_indices")
WEIGHTS = {"spike_f1": 0.40, "fit": 0.40, "whiteness": 0.10, "noise_realism": 0.10}
TOLERANCE_MS = 2


def _remove_drift(signal: np.ndarray, fs: float, window_s: float = 0.2) -> np.ndarray:
    window = int(round(window_s * fs)) // 2 * 2 + 1
    padded = np.pad(signal, window // 2, mode="edge")
    return signal - np.median(sliding_window_view(padded, window), axis=1)


def _mad_sigma(signal: np.ndarray) -> float:
    return float(np.median(np.abs(signal - np.median(signal))) / 0.6745)


def detect_spikes(
    signal: np.ndarray,
    fs: float,
    alpha: float = 4.5,
    dead_time_s: float = 0.008,
) -> np.ndarray:
    """Detect reference spike onsets using a robust threshold and dead time."""
    flat = _remove_drift(signal, fs)
    above = flat > alpha * _mad_sigma(flat)
    onsets = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    kept: list[int] = []
    for onset in onsets:
        if not kept or onset - kept[-1] >= dead_time_s * fs:
            kept.append(int(onset))
    return np.asarray(kept, dtype=int)


def noise_floor(signal: np.ndarray, spikes: np.ndarray, fs: float) -> float:
    """Estimate original noise SD from first differences away from spike tails."""
    keep = np.ones(len(signal) - 1, dtype=bool)
    for spike in spikes:
        keep[max(0, spike - 2): spike + int(0.020 * fs)] = False
    return _mad_sigma(np.diff(signal)[keep]) / np.sqrt(2)


def match_spikes(
    reference_indices: np.ndarray,
    candidate_indices: np.ndarray,
    tolerance_samples: int,
) -> tuple[float, float, float]:
    """Return one-to-one spike precision, recall, and F1."""
    used = np.zeros(len(reference_indices), dtype=bool)
    hits = 0
    for candidate in np.sort(candidate_indices):
        distances = np.abs(reference_indices - candidate)
        distances[used] = tolerance_samples + 1
        match = int(np.argmin(distances)) if len(distances) else -1
        if match >= 0 and distances[match] <= tolerance_samples:
            used[match] = True
            hits += 1
    precision = hits / len(candidate_indices) if len(candidate_indices) else 0.0
    recall = hits / len(reference_indices) if len(reference_indices) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def autocorr(signal: np.ndarray, lags: range = range(1, 6)) -> np.ndarray:
    centered = signal - signal.mean()
    denominator = np.dot(centered, centered)
    if denominator <= 0:
        return np.ones(len(list(lags)))
    return np.asarray(
        [np.dot(centered[:-lag], centered[lag:]) / denominator for lag in lags]
    )


def _load_candidate(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        candidate = {key: np.asarray(data[key]) for key in data.files}
    return candidate


def check_validity(
    reference_t: np.ndarray,
    candidate: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validate the standardized six-array candidate format."""
    n = len(reference_t)
    validity: dict[str, Any] = {
        "has_all_arrays": all(key in candidate for key in REQUIRED)
    }
    if not validity["has_all_arrays"]:
        validity["missing"] = [key for key in REQUIRED if key not in candidate]
        validity["valid"] = False
        return validity

    signal_keys = [key for key in REQUIRED if key != "spike_indices"]
    validity["lengths_ok"] = all(candidate[key].shape == (n,) for key in signal_keys)
    validity["time_matches"] = bool(
        validity["lengths_ok"] and np.allclose(candidate["t"], reference_t)
    )
    validity["all_finite"] = all(
        np.all(np.isfinite(candidate[key])) for key in REQUIRED
    )
    indices = np.asarray(candidate["spike_indices"])
    validity["spike_indices_ok"] = bool(
        indices.ndim == 1
        and np.all(indices == np.round(indices))
        and np.all((indices >= 0) & (indices < n))
        and np.all(np.diff(indices) > 0)
    )
    if validity["lengths_ok"]:
        structured = candidate["drift"] + candidate["spike_signal"]
        validity["reconstruction_is_drift_plus_spikes"] = bool(
            np.allclose(candidate["reconstruction"], structured, atol=1e-6)
        )
        validity["reconstruction_includes_noise"] = bool(
            np.allclose(candidate["reconstruction"], structured + candidate["noise"], atol=1e-6)
        )
    validity["valid"] = bool(
        validity["lengths_ok"]
        and validity["time_matches"]
        and validity["all_finite"]
        and validity["spike_indices_ok"]
    )
    return validity


def _overlay(
    time: np.ndarray,
    original: np.ndarray,
    structured: np.ndarray,
    candidate_indices: np.ndarray,
    reference_indices: np.ndarray,
    path: str | Path | None,
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(14, 6))
    for axis, (lower, upper) in zip(axes, ((0, len(time)), (300, 500))):
        selected = slice(lower, upper)
        axis.plot(time[selected], original[selected], lw=0.5, color="0.6", label="Original")
        axis.plot(
            time[selected],
            structured[selected],
            lw=1,
            color="tab:red",
            label="Candidate: drift + spikes",
        )
        reference = reference_indices[
            (reference_indices >= lower) & (reference_indices < upper)
        ]
        candidate = candidate_indices[
            (candidate_indices >= lower) & (candidate_indices < upper)
        ]
        axis.plot(time[reference], np.full(len(reference), original.max() + 0.3),
                  "kv", ms=5, label="Reference spikes")
        axis.plot(time[candidate], np.full(len(candidate), original.max() + 0.8),
                  "rv", ms=5, label="Candidate spikes")
        axis.set_xlabel("Time (s)")
    axes[0].legend(loc="lower left", fontsize=8, ncol=4)
    axes[0].set_title("Full recording")
    axes[1].set_title("Zoom: 0.30-0.50 s")
    figure.tight_layout()
    if path is not None:
        figure.savefig(path, dpi=120)
        plt.close(figure)
        return str(path)
    return figure


def evaluate(reference_file: str | Path, candidate_file: str | Path) -> dict[str, Any]:
    """Evaluate one candidate and return table-friendly results."""
    with np.load(reference_file) as reference:
        reference_t = np.asarray(reference["t"], dtype=float)
        reference_signal = np.asarray(reference["raw_data"], dtype=float)
    candidate = _load_candidate(candidate_file)
    validity = check_validity(reference_t, candidate)
    result: dict[str, Any] = {
        "reference": str(reference_file),
        "candidate": str(candidate_file),
        "validity": validity,
    }
    if not validity.get("valid", False):
        result.update(metrics={}, final_score=0.0, overlay_figure=None)
        return result

    fs = 1.0 / np.median(np.diff(reference_t))
    structured = candidate["drift"] + candidate["spike_signal"]
    leftover = reference_signal - structured
    reference_indices = detect_spikes(reference_signal, fs)
    candidate_indices = candidate["spike_indices"].astype(int)
    floor = noise_floor(reference_signal, reference_indices, fs)
    leftover_sd = float(np.std(leftover))
    precision, recall, f1 = match_spikes(
        reference_indices,
        candidate_indices,
        round(TOLERANCE_MS * fs / 1000),
    )
    correlations = autocorr(leftover)
    candidate_noise_sd = float(np.std(candidate["noise"]))
    parts = {
        "spike_f1": f1,
        "fit": min(floor / max(leftover_sd, 1e-12), leftover_sd / floor),
        "whiteness": max(
            0.0,
            1 - max(0.0, float(np.mean(np.abs(correlations))) - 2 / np.sqrt(len(reference_signal))) / 0.1,
        ),
        "noise_realism": max(0.0, 1 - abs(candidate_noise_sd / floor - 1) / 0.2),
    }
    final_score = 100 * sum(WEIGHTS[key] * parts[key] for key in WEIGHTS)
    result["metrics"] = {
        "n_spikes_reference": int(len(reference_indices)),
        "n_spikes_candidate": int(len(candidate_indices)),
        "spike_precision": round(precision, 4),
        "spike_recall": round(recall, 4),
        "noise_floor_sd": round(float(floor), 4),
        "leftover_sd": round(leftover_sd, 4),
        "leftover_autocorr_lags1to5": [round(float(value), 4) for value in correlations],
        "candidate_noise_sd": round(candidate_noise_sd, 4),
        "part_scores": {key: round(float(value), 4) for key, value in parts.items()},
        "weights": WEIGHTS,
    }
    result["final_score"] = round(float(final_score), 2)
    result["overlay_figure"] = _overlay(
        reference_t, reference_signal, structured,
        candidate_indices, reference_indices, None,
    )
    return result


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    separator = "-+-".join("-" * width for width in widths)
    print(f"\n{title}")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print(separator)
    for row in rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _display(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return "nan" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_file")
    parser.add_argument("candidate_file")
    args = parser.parse_args()
    result = evaluate(args.reference_file, args.candidate_file)

    _print_table(
        "Validity checks",
        ("Check", "Result"),
        [(key, _display(value)) for key, value in result["validity"].items()],
    )
    if result["metrics"]:
        metrics = result["metrics"]
        metric_rows = [
            ("Spike count", str(metrics["n_spikes_reference"]), str(metrics["n_spikes_candidate"]), ""),
            ("Spike precision", "", _display(metrics["spike_precision"]), ""),
            ("Spike recall", "", _display(metrics["spike_recall"]), ""),
            ("Noise floor SD", _display(metrics["noise_floor_sd"]), "", ""),
            ("Leftover SD", "", _display(metrics["leftover_sd"]), ""),
            ("Candidate noise SD", "", _display(metrics["candidate_noise_sd"]), ""),
            ("Whiteness autocorrelation", "", str(metrics["leftover_autocorr_lags1to5"]), ""),
        ]
        _print_table("Metric comparison", ("Metric", "Reference", "Candidate", "Distance"), metric_rows)
        _print_table(
            "Part scores",
            ("Component", "Score", "Weight", "Contribution"),
            [
                (key, _display(value), _display(WEIGHTS[key]), _display(value * WEIGHTS[key]))
                for key, value in metrics["part_scores"].items()
            ],
        )
    _print_table(
        "Final score",
        ("Score", "Value"),
        [("Overall score (0-100)", _display(result["final_score"]))],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
