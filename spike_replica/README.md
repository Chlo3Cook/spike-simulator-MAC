# spike_replica: reverse-engineering `spike_data_2.npz`

This project studies a 10-second voltage recording, works out how it was made, and generates new recordings of any length that look and behave like the original. It also includes a test and scoring module that measures how close a generated recording is to the real one.

---

## 1. What the project does

1. **Analyses** the real recording (`data/spike_data_2.npz`, keys `t` and `raw_data`, 10,000 samples at 1 kHz).
2. **Fits a model** with three parts: slow drift, spikes, and noise.
3. **Generates** new recordings of any length from the fitted model, saved in the same structure as the original file.
4. **Scores** how close the generated data is to the real data, with a table of 20 metrics and an overall score out of 100.

## 2. The model that was found

```
x(t) =  c + A_d * sin(2*pi*f*t + phase)              <- drift
      + sum over spikes k of  A_s * exp(-(t - t_k)/tau)   for t >= t_k   <- spikes
      + N(0, sigma^2)                                  <- white Gaussian noise
```

| Part | What it is | Fitted value |
|---|---|---|
| **Drift** | A slow sine wave (one cycle takes about 11 s) | frequency 0.0908 Hz, amplitude 0.647, phase 0.292 rad, offset about 0 |
| **Spikes** | Identical spikes that jump to a fixed height and decay exponentially. Spikes that overlap simply add. | 42 spikes (4.2 per second), amplitude 5.016, decay constant 2.983 ms |
| **Noise** | Independent Gaussian samples, with no correlation between neighbours | standard deviation 0.2985 |

After the drift and spikes are removed, what is left is white, Gaussian, and has constant variance, so nothing else is hiding in the signal. All fitted values are stored in `params/fitted_params.json`.

### Spike timing (0.5 s windows)

The fitted file also stores `window_counts`, the real number of spikes in each 0.5 s window of the 10 s recording:

```
[5, 1, 1, 3, 5, 3, 1, 0, 0, 4, 3, 4, 0, 2, 3, 2, 0, 2, 1, 2]     (total 42)
```

The generator uses these counts so that *when* spikes happen is close to the real recording:

- **First 10 s:** each 0.5 s window gets exactly the real count of spikes. The position of each spike inside its window is random, so the timing is close but not identical.
- **After 10 s:** each window's count is drawn at random from the real counts. This keeps the spike statistics of the real data, but the timing is new.
- **`--spike-mode poisson`:** switches this off. Spikes then occur at random times at the average rate, and no real counts are copied. Use this for a fully independent run.

---

## 3. Setup

You need Python 3.9 or newer.

```
pip install -r requirements.txt
```

This installs numpy, scipy, pytest and matplotlib. Matplotlib is only needed for the plot.

---

## 4. How to run it

Run every command from inside the `spike_replica` folder (the one that contains `Src/`, `data/` and `tests/`).

| Goal | Command |
|---|---|
| **Show the accuracy score table** | `python -m Src evaluate` |
| Run all tests (also prints the score table at the end) | `python -m pytest -q` |
| Re-extract the parameters from the real data | `python -m Src fit` |
| Generate a new recording (any length) | `python -m Src generate --duration 120 --seed 7 --out results/generated_data.npz` |
| Score an existing `.npz` file against the real data | `python -m Src evaluate --generated results/generated_data.npz` |
| Score the model over 20 different random seeds | `python -m Src evaluate --seeds 20` |
| Save the comparison figure | `python -m Src evaluate --plot results/comparison.png` |
| Compare against plain random (Poisson) spike timing | `python -m Src evaluate --spike-mode poisson` |

Options for `generate`:

- `--duration` is the length in seconds (default 60).
- `--seed` is the random seed. The same seed always gives the same recording.
- `--spike-mode window` or `poisson` (default `window`).
- `--out` is the output file path.


## 5. What is in each file

```
spike_replica/
├── README.md               this file
├── requirements.txt        Python packages needed
├── pytest.ini              tells pytest to look in tests/
├── conftest.py             makes `import Src` work in tests; prints the score table after pytest
├── data/
│   └── spike_data_2.npz    the real recording (keys: t, raw_data)
├── params/
│   └── fitted_params.json  fitted drift, spike and noise values + the 0.5 s window counts
├── results/
│   ├── generated_data.npz  a 60 s generated recording (same structure as the original file)
│   ├── score_report.txt    the saved score table
│   └── comparison.png      real vs generated figure (see section 8)
├── Src/                    the code
│   ├── __init__.py         the public interface of the package
│   ├── __main__.py         lets you run `python -m Src ...`
│   ├── cli.py              the command line: fit / generate / evaluate
│   ├── io.py               read and write .npz files with keys t and raw_data
│   ├── model.py            Params (all fitted numbers), the drift, the spike shape, window counting
│   ├── fit.py              spike detection and the least-squares fit of all parameters
│   ├── generate.py         the simulator: drift + spikes + noise for any duration
│   ├── metrics.py          the 20 accuracy metrics and the scoring rule
│   └── evaluate.py         prints the score table and draws the comparison figure
└── tests/
    ├── test_generate.py    output format, seeds, spike shape, 0.5 s window counts
    ├── test_fit.py         recovers known parameters from synthetic data; results on the real data
    └── test_accuracy.py    generated data vs real data, and a check that wrong models score lower
```

### How the pieces work

- **`model.py`** holds the `Params` dataclass, which is every number the model needs. It also defines the drift curve and the spike signal. The exponential spike is built with a one-line recursive filter, and overlapping spikes add automatically. `window_counts()` counts spikes per 0.5 s window.
- **`fit.py`** goes through these steps:
  1. Find rough spike onsets: a sample counts as an onset when it jumps up by more than 8 robust noise standard deviations.
  2. Search for the drift frequency (a least-squares fit at each trial frequency).
  3. Refine all parameters together.
  4. Look in the leftover signal for spikes that hid inside another spike's tail, add them, and refit. This repeats until nothing new turns up.
  5. Compute the noise level, the spike rate and the window counts.
- **`generate.py`** picks the spike times (window mode or Poisson mode), then adds drift, spikes and Gaussian noise.
- **`metrics.py`** runs the same fit on the real data and the generated data, so each comparison is like for like.
- **`evaluate.py`** prints the table, the window counts side by side, and optionally the figure.

---

## 6. How the score works

Both recordings go through the same fitting code, then 20 metrics are compared, grouped as follows:

| Group | Metrics |
|---|---|
| **drift** (4) | frequency, amplitude, offset, difference between the two drift curves |
| **spike** (5) | rate, amplitude, decay time, average spike waveform, inter-spike-interval distribution |
| **timing** (4) | mean spikes per 0.5 s window, Fano factor (how bursty the counts are), histogram of counts per window, window-by-window count error over the first 10 s |
| **noise** (5) | standard deviation, distribution shape, skew and kurtosis, whiteness (no correlation), power spectrum of the noise |
| **signal** (2) | power spectrum of the whole signal, amplitude distribution without the drift |

Each metric has an **error** (0 means identical) and a **tolerance**:

```
score  = 100 * max(0, 1 - error / tolerance)
PASS   = score >= 50
overall = average of all metric scores
```

The generated recording can be any length. It is compared with the 10 s real recording using statistics that do not depend on length.

### Current result

A fresh 60 s generation scores **91.0 / 100** with 20 of 20 checks passing. Over 20 different seeds the overall score ranged from 82.8 to 91.8 (mean 88.9). The window counts of the first 10 s match the real data in every window.

---

## 7. Limits

- **A generated file cannot match the original sample by sample.** The noise and the exact spike positions are random. The best possible pointwise R² for any model is about **0.835** (= 1 − noise variance / signal variance). The score table prints this as an info line and does not score it. The drift, the spike shape and the spike counts per window do match.
- **The real file is short.** It holds only 42 spikes and about 0.9 of a drift cycle, so some comparisons carry sampling error. Even a perfect model scores about 85 to 92 overall for that reason, not 100.
- **The drift is extrapolated** as one sine wave. This matches the data, but 10 s of data cannot rule out slower components.
- **The first 10 s of window counts are copied from the real data** by design. Use `--spike-mode poisson` for a fully independent test.

## 8. About `comparison.png`

The figure has four rows:

1. The real 10 s recording and the first 10 s of the generated recording, side by side on the same axes.
2. The full generated recording (60 s), with the first 10 s shaded.
3. Spikes per 0.5 s window, real next to generated.
4. The power spectrum and the amplitude histogram, with real and generated overlaid so that differences are easy to see.
