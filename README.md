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

## Source folder

### [`src/simulator.ipynb`](./src/simulator.ipynb)

The main analysis and generation notebook. It loads the reference data, fits
the model, checks the fit, creates figures, defines `generate_signal()`, and
can save generated recordings in the same format as the input:

```python
np.savez("simulated_spike_data.npz", t=t, raw_data=signal)
```

The saved file contains:

- `t`: time values in seconds
- `raw_data`: the complete simulated voltage signal

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

### Example comparison result

The following result compares a 10-second simulation generated with
`seed=None` against the 10-second reference recording. With `seed=None`, the
spike times and noise are newly randomized each run:

| Metric | Original | Simulated | Spreads away | Meaning / significance |
|---|---:|---:|---:|---|
| `noise_sd` | 0.311 | 0.312 | 0.17 | Measures random background noise; the nearly identical values indicate a well-matched noise level. |
| `drift_sd` | 0.484 | 0.477 | 1.62 | Measures slow baseline variation; the small difference indicates similar drift strength. |
| `n_spikes` | 42.000 | 43.000 | 0.16 | Counts detected spikes; the nearly identical counts indicate a comparable firing amount. |
| `mean_height` | 4.978 | 4.308 | 12.64 | Measures average spike size; the large difference indicates that simulated spikes are shorter or lower on average. |
| `isi_cv` | 1.074 | 0.904 | 1.27 | Measures variability in the time between spikes; the difference indicates somewhat different firing regularity. |
| **Overall score** |  |  | **5.73** | Combined root-mean-square distance; lower is better. |

The simulation matches the original well in noise level, drift strength, and
spike count. The main mismatch is mean spike height, followed by inter-spike
interval variability. The score of `5.73` is above the approximate `1.4`
value expected for two recordings of the same neuron, so this run is
statistically similar in several properties but is not yet a close overall
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
| Drift | Offset, `c` | 0.0223 | Matches the fitted average voltage level. |
| Drift | Amplitude, `B` | 0.651 | Matches the observed size of the slow baseline variation. |
| Drift | Frequency, `f` | 0.0899 Hz | Gives a period of approximately 11.1 seconds, consistent with the recording’s slow drift. |
| Drift | Phase, `phi` | 0.32 rad | Aligns the sinusoidal drift with the observed recording. |
| Spikes | Model | `exp_fixed` | The spikes are well represented by a fast jump followed by exponential decay, using one shared shape for all spikes. |
| Spikes | Amplitude, `A` | 4.96 | Matches the fitted spike height. |
| Spikes | Decay constant, `tau` | 2.86 ms | Matches the observed spike duration and falling edge. |
| Timing | Model | `poisson_n_gap` | Produces random spike locations while preserving the observed average firing behavior and preventing unrealistic overlaps. |
| Timing | Average rate | 4.20 Hz | Estimated from approximately 42 spikes in the 10-second reference recording. |
| Timing | Minimum gap | 9.9 ms | Reflects the shortest observed gap and avoids placing spikes unrealistically close together. |
| Noise | Model | `white` | Independent Gaussian noise was sufficient to reproduce the recording’s residual variation. |
| Noise | Standard deviation, `sigma` | 0.293 | Fitted from the signal after removing drift and spike structure. |

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
