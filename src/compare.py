"""Testing module: how close is a simulated recording to the original?"""
import numpy as np
from spike_detection import remove_drift, estimate_noise, detect_spikes

REQUIRED_KEYS = ("t", "raw_data")


def load_recording(path):
    """Load an .npz file and return (t, raw_data). Raise an error if the keys or shapes are wrong."""
    data = np.load(path)
    for key in REQUIRED_KEYS:
        if key not in data.files:
            raise ValueError(f"{path} is missing '{key}' (it contains {data.files})")
    t = np.asarray(data["t"], dtype=float)
    x = np.asarray(data["raw_data"], dtype=float)
    if t.ndim != 1 or x.ndim != 1 or len(t) != len(x):
        raise ValueError(f"{path}: 't' and 'raw_data' must be 1-D and the same length")
    return t, x


def check_structure(t_orig, t_sim):
    """Return a True/False for each structural property the simulation must share with the original."""
    return {
        "same_length": len(t_orig) == len(t_sim),
        "same_start": bool(np.isclose(t_orig[0], t_sim[0])),
        "same_sampling": bool(np.isclose(np.median(np.diff(t_orig)), np.median(np.diff(t_sim)))),
    }


def summarize(t, x):
    """Measure the properties of one recording that a good simulation must reproduce."""
    fs = 1 / np.median(np.diff(t))      # sampling rate from the time array
    flat = remove_drift(x, fs)          # signal with the slow drift removed
    baseline = x - flat                 # the slow drift itself
    spikes = detect_spikes(x, fs)       # sample index of each spike onset
    gaps = np.diff(spikes) / fs         # time between spikes, in seconds
    return {
        "noise_sd": float(estimate_noise(flat)),
        "drift_sd": float(np.std(baseline)),
        "n_spikes": len(spikes),
        "mean_height": float(np.mean(flat[spikes])) if len(spikes) else np.nan,
        "isi_cv": float(np.std(gaps) / np.mean(gaps)) if len(gaps) > 1 else np.nan,
    }
# How much each measurement varies naturally between two recordings of the same neuron.
# Standard deviation over 500 simulated 10 s recordings (fitted model, Poisson spike times).
# Recompute with estimate_spread() below.
NATURAL_SPREAD = {
    "noise_sd": 0.0042,
    "drift_sd": 0.0046,
    "n_spikes": 6.08,
    "mean_height": 0.053,
    "isi_cv": 0.134,
}


def estimate_spread(n_runs=500, duration=10):
    """Recompute NATURAL_SPREAD: simulate n_runs recordings with the reference model,
    measure each one, and return the standard deviation of every measurement."""
    from reference_simulator import simulate
    rows = []
    for seed in range(n_runs):
        t, x, _ = simulate(duration, seed=seed)
        rows.append(summarize(t, x))
    return {key: round(float(np.nanstd([row[key] for row in rows])), 4) for key in rows[0]}


def score(orig, sim):
    """How far each simulated measurement is from the original, in units of natural spread."""
    distances = {key: abs(sim[key] - orig[key]) / NATURAL_SPREAD[key] for key in NATURAL_SPREAD}
    overall = float(np.sqrt(np.mean(np.square(list(distances.values())))))   # root mean square: big misses count more
    return distances, overall


def summarize_generated(generate_signal, duration, n_runs=500):
    """Average measurements from n_runs generated recordings.

    generate_signal must accept duration_s and seed keyword arguments and
    return the generated time array and voltage array as its first two values.
    """
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    rows = []
    structures = []
    for seed in range(n_runs):
        generated = generate_signal(duration_s=duration, seed=seed)
        if len(generated) < 2:
            raise ValueError("generate_signal must return time and voltage arrays")
        t, x = generated[:2]
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        if t.ndim != 1 or x.ndim != 1 or len(t) != len(x):
            raise ValueError("generate_signal returned invalid time and voltage arrays")
        structures.append((t, x))
        rows.append(summarize(t, x))

    metrics = rows[0].keys()
    mean_metrics = {}
    for key in metrics:
        values = [row[key] for row in rows if np.isfinite(row[key])]
        mean_metrics[key] = float(np.mean(values)) if values else np.nan
    return mean_metrics, structures


def compare_generated(original_path, generate_signal, n_runs=500):
    """Compare the original recording with the mean of generated metrics."""
    t_orig, x_orig = load_recording(original_path)
    duration = float(t_orig[-1] + np.median(np.diff(t_orig)))
    orig = summarize(t_orig, x_orig)
    sim, generated = summarize_generated(generate_signal, duration, n_runs)
    structure = check_structure(t_orig, generated[0][0])
    distances, overall = score(orig, sim)

    print("Structure:", structure)
    print(f"Compared {n_runs} generated recordings; simulated values are metric means.")
    print(f"{'measure':<12} {'original':>9} {'simulated':>10} {'spreads away':>13}")
    for key in orig:
        print(f"{key:<12} {orig[key]:>9.3f} {sim[key]:>10.3f} {distances[key]:>13.2f}")
    print(f"OVERALL SCORE: {overall:.2f}   (lower is closer; about 1.4 = as close as a second recording of the same neuron)")
    return structure, sim, distances, overall


def compare_files(original_path, simulated_path):
    """Load both files, check structure, measure both, score the simulation, and print a report."""
    t_o, x_o = load_recording(original_path)
    t_s, x_s = load_recording(simulated_path)
    structure = check_structure(t_o, t_s)
    orig = summarize(t_o, x_o)
    sim = summarize(t_s, x_s)
    distances, overall = score(orig, sim)

    print("Structure:", structure)
    print(f"{'measure':<12} {'original':>9} {'simulated':>10} {'spreads away':>13}")
    for key in orig:
        print(f"{key:<12} {orig[key]:>9.3f} {sim[key]:>10.3f} {distances[key]:>13.2f}")
    print(f"OVERALL SCORE: {overall:.2f}   (lower is closer; about 1.4 = as close as a second recording of the same neuron)")
    return structure, distances, overall


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("Usage: python src/compare.py ORIGINAL.npz SIMULATED.npz")
    compare_files(sys.argv[1], sys.argv[2])
