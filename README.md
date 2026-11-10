# Spike Simulator

This repository models and evaluates synthetic neural-voltage recordings from
the reference recording in [`data/spike_data.npz`](./data/spike_data.npz).

## Workflow

1. Use [`spike_model_search.ipynb`](./spike_model_search.ipynb) to compare
   candidate models.
2. Use [`simulator.ipynb`](./simulator.ipynb) to fit the selected
   model and generate a standardized reconstruction.
3. Use [`evaluate_team.py`](./evaluate_team.py) to evaluate a reconstruction.

The generated recording is statistically similar to the original. It is not
expected to reproduce the same sample-by-sample voltage or spike locations.

## Selected model

The model-search notebook compares baseline drift, spike shape, spike timing,
and noise models. The selected combination is:

- `sin1` for the slow drift
- `exp_fixed` for the spike waveform
- `poisson_n_gap` for spike timing
- `white` for measurement noise

The signal is modeled as:

```text
signal(t) = drift(t) + sum of spike(t - t_i) + noise(t)
```

## Reconstruction

The standardized reconstruction is
[`team_05_reconstruction.npz`](./team_05_reconstruction.npz). It contains:

- `t`: the 10,000-sample time vector in seconds
- `drift`: the fitted slow baseline component
- `spike_signal`: the fitted sum of exponential spike waveforms
- `noise`: a generated Gaussian noise realization
- `reconstruction`: the structured reconstruction, `drift + spike_signal`
- `spike_indices`: the detected spike starting indices

The `reconstruction` array intentionally excludes the generated noise
realization.

## Evaluation

[`evaluate_team.py`](./evaluate_team.py) implements the
`evaluate(reference_file, candidate_file)` interface. It checks the candidate
format and calculates spike timing, fit, whiteness, and noise-realism scores.
The result includes validity checks, metric values, a Matplotlib overlay, and a
final score from 0 to 100.

Run it from the repository root:

```powershell
python evaluate_team.py data\spike_data.npz team_05_reconstruction.npz
```

## Simulator notebook

[`simulator.ipynb`](./simulator.ipynb) loads the reference data, fits
the model, checks the fit, creates figures, defines `generate_signal()`, and
saves reconstructions in the standardized format:

```python
np.savez(
    "team_05_reconstruction.npz",
    t=t,
    drift=drift_all,
    spike_signal=spike_component,
    noise=noise,
    reconstruction=drift_all + spike_component,
    spike_indices=onset_indices,
)
```

For a newly generated noisy recording, use
`generate_signal(duration_s, seed)`, which returns `(time, voltage)`.

## Model parameters

The final fit to the 10-second reference recording produced:

| Parameter | Value |
|---|---:|
| Sampling rate | 1000 Hz |
| Drift offset | 0.003002 |
| Drift amplitude | 0.648357 |
| Drift frequency | 0.090823 Hz |
| Drift phase | 0.286174 rad |
| Spike amplitude | 5.016163 |
| Spike decay | 2.9820 ms |
| Noise standard deviation | 0.298455 |
| Detected spike count | 42 |

For longer simulations, the fitted drift and waveform parameters are retained,
while new spike times and noise are generated using the selected timing and
noise models.

## Analysis method

1. Estimate the slow baseline with a centered running median.
2. Subtract the baseline and estimate robust noise using the median absolute
   deviation.
3. Detect spike onsets from threshold crossings in the detrended signal.
4. Fit the sinusoidal drift by frequency grid search and least squares.
5. Fit a shared exponential spike waveform using the detected onsets.
6. Reconstruct the structured signal and estimate the residual noise level.

## Setup

Create and activate a virtual environment, then install the packages listed in
[`requirements.txt`](./requirements.txt):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The reference recording contains 10,000 samples at a 1 kHz sampling rate,
corresponding to 10 seconds of data.
