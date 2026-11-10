# Spike Simulator

This repository builds and evaluates synthetic neural-voltage recordings from
the reference recording in `data/spike_data.npz`.

## Model selection

[`spike_model_search_(1).ipynb`](./spike_model_search_(1).ipynb) was used to
select the best generative model. It tests combinations for the baseline
drift, spike shape, spike timing, and noise against the original 10-second
recording. The selected combination is:

- `sin1` for the slow drift
- `exp_fixed` for the spike waveform
- `poisson_n_gap` for spike timing
- `white` for measurement noise

The final model is a sum of these components:

```text
signal(t) = drift(t) + sum of spike(t - t_i) + noise(t)
```

The generated recording is statistically similar to the original; it is not
expected to reproduce the same sample-by-sample voltage or spike locations.

## Part 1 submission

The standardized submission is
[`team_05_reconstruction.npz`](./team_05_reconstruction.npz). It contains:

- `t`: the 10,000-sample time vector in seconds
- `drift`: the fitted slow baseline component
- `spike_signal`: the fitted sum of exponential spike waveforms
- `noise`: a Gaussian noise realization generated with seed `42`
- `reconstruction`: the structured reconstruction `drift + spike_signal`
- `spike_indices`: the 42 detected spike starting indices

The `reconstruction` array intentionally excludes the newly generated noise
realization, as required by the task.

## Part 2 evaluator

[`evaluate_team.py`](./evaluate_team.py) implements the required
`evaluate(reference_file, candidate_file)` interface. It accepts the
standardized `reconstruction` array and also accepts legacy `raw_data`
candidates. The returned dictionary contains:

- validity and structure checks
- reference and candidate values for the metrics from `src/compare.py`
- normalized metric distances
- a Matplotlib overlay figure
- `final_score`, the required overall RMS distance

Run it from the repository root with:

```powershell
python evaluate_team.py data\spike_data.npz team_05_reconstruction.npz
```

## Inferred model and parameters

The recording is modeled as a sinusoidal drift, a shared exponential spike
waveform at detected onset times, and independent Gaussian measurement noise:

```text
x(t) = c + B sin(2*pi*f*t + phi)
       + sum_k A exp(-(t - t_k)/tau) 1[t >= t_k]
       + Normal(0, sigma^2)
```

The final fit to all 10 seconds produced:

| Parameter | Value |
|---|---:|
| Sampling rate | 1000 Hz |
| Drift offset, `c` | 0.003002 |
| Drift amplitude, `B` | 0.648357 |
| Drift frequency, `f` | 0.090823 Hz |
| Drift phase, `phi` | 0.286174 rad |
| Spike amplitude, `A` | 5.016163 |
| Spike decay, `tau` | 2.9820 ms |
| Noise SD, `sigma` | 0.298455 |
| Detected spike count | 42 |

## Assumptions

- The slow baseline can be approximated by one sinusoid over the 10-second
  recording.
- All spikes share one positive exponential waveform and amplitude.
- Detected onset times are treated as the spike times for reconstruction.
- Residual measurement noise is independent, Gaussian, and has constant
  variance.
- The recorded signal is sufficient to estimate structure, but the hidden
  original noise and generative spike timing process are not observable.

## Detection and reconstruction method

1. Estimate the slow baseline with a centered 101 ms running median.
2. Subtract that baseline and estimate robust noise using the median absolute
   deviation.
3. Mark a spike onset at the first sample crossing eight robust noise standard
   deviations above the detrended signal. This detects 42 onsets.
4. Fit the sinusoidal drift by frequency grid search and least squares, excluding
   detected spike transients.
5. Fit one shared exponential decay constant and amplitude by grid search and
   least squares using the detected onset times.
6. Reconstruct the structured signal as `drift + spike_signal`; estimate the
   residual noise level from the difference between the recording and this
   structured fit.

## Source folder

### [`src/simulator.ipynb`](./src/simulator.ipynb)

The main analysis and generation notebook. It loads the reference data, fits
the model, checks the fit, creates figures, defines `generate_signal()`, and
can save generated recordings in the same format as the input:

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

For a newly generated noisy recording, use `generate_signal(duration_s, seed)`,
which returns `(time, voltage)`.

### [`src/reference_simulator.py`](./src/reference_simulator.py)

A small reusable reference implementation. It generates a signal from
sinusoidal drift, Gaussian noise, exponentially decaying spikes, and Poisson
spike times:

```python
from reference_simulator import simulate

t, raw_data, spike_times = simulate(duration=30, seed=42)
```

### [`src/spike_detection.py`](./src/spike_detection.py)

Signal-processing utilities for removing slow drift, estimating noise,
detecting spike onsets, and applying a dead time so one spike is not counted
multiple times.

### [`src/waveforms.py`](./src/waveforms.py)

Extracts short waveform windows around detected spikes and aligns each window
to its local peak.

### [`src/compare.py`](./src/compare.py)

Compares an original `.npz` recording with a simulated `.npz` recording. It
checks recording structure and compares noise, drift, spike count, spike
height, and inter-spike-interval variation.

Run it from the repository root with:

```powershell
python src\compare.py data\spike_data.npz simulated_spike_data.npz
```

The overall score is measured in units of natural recording-to-recording
variation; lower scores mean the recordings are more similar.

To compare the original recording with the mean metrics from 500 generated
recordings in `src/simulator.ipynb`, run the notebook's generator definition
first, then add and run:

```python
import sys
sys.path.insert(0, "src")
from compare import compare_generated

compare_generated("data/spike_data.npz", generate_signal, n_runs=500)
```

Each run uses a different deterministic seed (`0` through `499`), and the
reported simulated values are the arithmetic means of the metrics across all
500 runs. This reduces the influence of one random recording while preserving
reproducibility.

### Example comparison result

The following result compares the 10-second reference recording with the
mean metrics from 500 generated recordings. 

| Metric | Original | Simulated mean | Spreads away | Meaning / significance |
|---|---:|---:|---:|---|
| `noise_sd` | 0.311 | 0.310 | 0.45 | Measures random background noise; the close mean indicates a well-matched noise level. |
| `drift_sd` | 0.484 | 0.481 | 0.64 | Measures slow baseline variation; the close mean indicates similar drift strength. |
| `n_spikes` | 42.000 | 42.078 | 0.01 | Counts detected spikes; the almost identical mean indicates a matched firing amount. |
| `mean_height` | 4.978 | 4.226 | 14.19 | Measures average spike size; the large difference is the main mismatch. |
| `isi_cv` | 1.074 | 0.923 | 1.13 | Measures variability in inter-spike timing; the moderate difference indicates somewhat different firing regularity. |
| **Overall score** |  |  | **6.38** | Combined root-mean-square distance; lower is better. |

The 500-run average matches the original well in noise level, drift strength,
and spike count. Mean spike height remains the main mismatch, while
inter-spike timing is closer. The overall score of `6.38` is above the
approximate `1.4` value expected for two recordings of the same neuron, so the
model reproduces several statistical properties but is not yet a close overall
match.

## Data

[`data/spike_data.npz`](./data/spike_data.npz) is the reference recording. It
contains 10,000 samples at a 1 kHz sampling rate, corresponding to 10 seconds
of data.

## Selected parameters

The final values were fitted from the reference recording and selected because
the model-search notebook found this combination to reproduce the observed
signal statistics while remaining simple and interpretable.

| Component | Parameter | Selected value | Reason |
|---|---|---:|---|
| Sampling | Sampling rate | 1000 Hz | Matches the time array in the reference recording and gives 1 ms samples. |
| Drift | Model | `sin1` | A single sinusoid captured the slow baseline movement with fewer assumptions than a more flexible model. |
| Drift | Offset, `c` | 0.003002 | Fitted from the all-data recording. |
| Drift | Amplitude, `B` | 0.648357 | Fitted from the all-data recording. |
| Drift | Frequency, `f` | 0.090823 Hz | Gives a period of approximately 11.0 seconds. |
| Drift | Phase, `phi` | 0.286174 rad | Aligns the sinusoidal drift with the recording. |
| Spikes | Model | `exp_fixed` | The spikes are well represented by a fast jump followed by exponential decay, using one shared shape for all spikes. |
| Spikes | Amplitude, `A` | 5.016163 | Matches the fitted spike height. |
| Spikes | Decay constant, `tau` | 2.9820 ms | Matches the observed spike duration and falling edge. |
| Timing | Model | `poisson_n_gap` | Produces random spike locations while preserving the observed average firing behavior and preventing unrealistic overlaps. |
| Timing | Average rate | 4.20 Hz | Estimated from approximately 42 spikes in the 10-second reference recording. |
| Timing | Minimum gap | 10 ms | Prevents generated spikes from overlapping unrealistically. |
| Noise | Model | `white` | Independent Gaussian noise was sufficient to reproduce the recording’s residual variation. |
| Noise | Standard deviation, `sigma` | 0.298455 | Fitted from the residual after removing drift and spike structure. |

For longer simulations, the fitted drift and waveform parameters are retained,
while new spike times and noise are randomly generated using the selected
timing and noise models.

## Setup

Create and activate a standard Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the tests with:

```powershell
pytest
```
