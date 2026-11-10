"""Evaluate a team's spike reconstruction against the reference recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from compare import check_structure, load_recording, score, summarize


def _load_candidate(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, str, np.ndarray | None]:
    """Load a candidate using its standardized reconstruction or legacy raw_data."""
    candidate_path = Path(path)
    with np.load(candidate_path) as data:
        if "t" not in data.files:
            raise ValueError(f"{candidate_path} is missing 't'")

        signal_key = "reconstruction" if "reconstruction" in data.files else "raw_data"
        if signal_key not in data.files:
            raise ValueError(
                f"{candidate_path} must contain 'reconstruction' or 'raw_data'; "
                f"found {data.files}"
            )

        t = np.asarray(data["t"], dtype=float)
        signal = np.asarray(data[signal_key], dtype=float)
        spike_indices = (
            np.asarray(data["spike_indices"], dtype=int)
            if "spike_indices" in data.files
            else None
        )

    if t.ndim != 1 or signal.ndim != 1 or t.shape != signal.shape:
        raise ValueError(
            f"{candidate_path}: 't' and '{signal_key}' must be 1-D and the same length"
        )
    if len(t) < 2:
        raise ValueError(f"{candidate_path}: at least two samples are required")
    if not np.isfinite(t).all() or not np.isfinite(signal).all():
        raise ValueError(f"{candidate_path}: 't' and '{signal_key}' must be finite")
    if not np.all(np.diff(t) > 0):
        raise ValueError(f"{candidate_path}: 't' must be strictly increasing")

    if spike_indices is not None:
        if spike_indices.ndim != 1:
            raise ValueError(f"{candidate_path}: 'spike_indices' must be one-dimensional")
        if np.any(spike_indices < 0) or np.any(spike_indices >= len(t)):
            raise ValueError(f"{candidate_path}: 'spike_indices' contains invalid positions")
        if np.any(np.diff(spike_indices) <= 0):
            raise ValueError(f"{candidate_path}: 'spike_indices' must be strictly increasing")

    return t, signal, signal_key, spike_indices


def _overlay_figure(
    reference_t: np.ndarray,
    reference_signal: np.ndarray,
    candidate_t: np.ndarray,
    candidate_signal: np.ndarray,
) -> Any:
    """Create an overlay figure without changing either input array."""
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(reference_t, reference_signal, linewidth=0.6, alpha=0.7, label="Reference")
    axis.plot(
        candidate_t,
        candidate_signal,
        linewidth=0.6,
        alpha=0.7,
        label="Candidate reconstruction",
    )
    axis.set(
        xlabel="Time (s)",
        ylabel="Voltage (a.u.)",
        title="Reference recording versus candidate reconstruction",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    return figure


def evaluate(reference_file: str | Path, candidate_file: str | Path) -> dict[str, Any]:
    """Evaluate a candidate reconstruction against the reference recording.

    Parameters
    ----------
    reference_file:
        Path to the original ``spike_data.npz`` containing ``t`` and ``raw_data``.
    candidate_file:
        Path to a candidate containing ``t`` and ``reconstruction``. Legacy
        candidates containing ``t`` and ``raw_data`` are also accepted.

    Returns
    -------
    dict
        Validity checks, metric summaries, the metric distances, an overlay
        figure, and the required ``final_score``.
    """
    reference_t, reference_signal = load_recording(reference_file)
    candidate_t, candidate_signal, signal_key, spike_indices = _load_candidate(candidate_file)

    structure = check_structure(reference_t, candidate_t)
    reference_metrics = summarize(reference_t, reference_signal)
    candidate_metrics = summarize(candidate_t, candidate_signal)
    if spike_indices is not None:
        candidate_metrics["n_spikes"] = len(spike_indices)
    distances, final_score = score(reference_metrics, candidate_metrics)

    validity = {
        "reference_file": str(reference_file),
        "candidate_file": str(candidate_file),
        "candidate_signal_key": signal_key,
        "candidate_spike_indices": spike_indices is not None,
        "candidate_arrays_valid": True,
        **structure,
    }

    return {
        "validity": validity,
        "metrics": {
            "reference": reference_metrics,
            "candidate": candidate_metrics,
            "distances": distances,
        },
        "overlay_figure": _overlay_figure(
            reference_t, reference_signal, candidate_t, candidate_signal
        ),
        "final_score": final_score,
    }


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    """Print aligned plain-text table rows for command-line use."""
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


def _display_value(value: Any) -> str:
    """Format metric values consistently, including missing numeric values."""
    if isinstance(value, (float, np.floating)):
        return "nan" if not np.isfinite(value) else f"{value:.4f}"
    return str(value)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_file", help="Path to the original spike_data.npz")
    parser.add_argument("candidate_file", help="Path to a candidate reconstruction")
    args = parser.parse_args()

    result = evaluate(args.reference_file, args.candidate_file)
    validity_rows = [
        (key, _display_value(value)) for key, value in result["validity"].items()
    ]
    _print_table("Validity checks", ("Check", "Result"), validity_rows)

    metrics = result["metrics"]
    metric_rows = [
        (
            key,
            _display_value(metrics["reference"][key]),
            _display_value(metrics["candidate"][key]),
            _display_value(metrics["distances"][key]),
        )
        for key in metrics["reference"]
    ]
    _print_table(
        "Metric comparison",
        ("Metric", "Reference", "Candidate", "Distance"),
        metric_rows,
    )
    _print_table(
        "Final score",
        ("Score", "Value"),
        [("Overall RMS distance", _display_value(result["final_score"]))],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
