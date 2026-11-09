"""Score table and comparison plot: how close is a generated recording to the real one?

    python -m Src evaluate                                           # generate fresh data, score it
    python -m Src evaluate --generated results/generated_data.npz    # score an existing file
"""
import numpy as np

from .generate import generate
from .io import load_npz
from .metrics import compare, overall, oracle_r2
from .model import Params, window_counts


def table(metrics):
    """Format the metrics as an aligned text table."""
    w = max(len(m.name) for m in metrics)
    head = f"{'group':<7}{'metric':<{w+2}}{'real':>10}{'generated':>11}{'error':>9}{'tol':>7}{'score':>7}  result"
    lines = [head, "-" * len(head)]
    for m in metrics:
        lines.append(f"{m.group:<7}{m.name:<{w+2}}{m.real:>10.4g}{m.gen:>11.4g}{m.error:>9.3g}{m.tol:>7.2g}"
                     f"{m.score:>7.1f}  {'PASS' if m.passed else 'FAIL'}")
    return "\n".join(lines)


def run(real_path, params_path, generated_path=None, duration=60.0, seed=0, n_seeds=0, plot=None,
        spike_mode="window", quiet=False):
    """Score a generated recording against the real one. Returns the overall score (0-100).

    generated_path - score this file; if None a fresh recording is generated from the fitted parameters
    n_seeds        - also report the score spread over this many different generator seeds
    plot           - path of a PNG to write (side-by-side comparison figure)
    """
    tr, xr = load_npz(real_path)
    P = Params.load(params_path)
    if generated_path:                                       # score a file produced elsewhere
        tg, xg = load_npz(generated_path)
        label = f"file {generated_path}"
    else:                                                    # generate now
        tg, xg, _ = generate(duration, P, seed=seed, spike_mode=spike_mode)
        label = f"fresh generation ({duration:g} s, seed {seed}, spike mode '{spike_mode}')"

    metrics, (pr, ir), (pg, ig) = compare(tr, xr, tg, xg)
    score, groups = overall(metrics)

    out = [f"Real: {len(xr)} samples ({len(xr)/pr.fs:.1f} s)   Generated: {len(xg)} samples ({len(xg)/pr.fs:.1f} s) -- {label}",
           "", table(metrics), ""]
    out.append("Group scores: " + "  ".join(f"{g}={s:.1f}" for g, s in groups.items()))
    out.append(f"Checks passed: {sum(m.passed for m in metrics)}/{len(metrics)}")
    out.append(f"OVERALL SIMILARITY SCORE: {score:.1f} / 100")

    # show the spike counts per 0.5 s window side by side (real vs the first 10 s of the generated data)
    nw = int(len(xr) / pr.fs // pr.window_s)
    cr = window_counts(ir["onsets"], len(xr), pr.fs, pr.window_s)[:nw]
    cg = window_counts(ig["onsets"], len(xg), pg.fs, pg.window_s)[:nw]
    out.append(f"\nSpikes per {pr.window_s:g} s window, first {nw*pr.window_s:g} s:\n  real      {cr.tolist()}\n  generated {cg.tolist()}")

    # pointwise ceiling (information only)
    r2 = oracle_r2(xr, pr, ir["onsets"])
    out.append(f"\n[info, not scored] Pointwise R^2 of the noise-free model (fitted drift + spikes at real spike times) vs real: {r2:.3f}"
               f"\n  (noise is random, so this is the highest pointwise R^2 any model can reach: 1 - noise_var/signal_var)")

    if n_seeds:                                              # robustness: many different random generations
        sc = [overall(compare(tr, xr, *generate(duration, P, seed=s, spike_mode=spike_mode)[:2],
                              real_fit=(pr, ir))[0])[0] for s in range(n_seeds)]
        out.append(f"\nRobustness over {n_seeds} generator seeds ({duration:g} s each): "
                   f"mean {np.mean(sc):.1f}, min {np.min(sc):.1f}, max {np.max(sc):.1f}")
    if plot:
        _plot(tr, xr, tg, xg, pr.fs, pr.window_s, ir["onsets"], ig["onsets"], plot)
        out.append(f"\nPlot saved to {plot}")
    if not quiet:
        print("\n".join(out))
    return score


def _plot(tr, xr, tg, xg, fs, win, on_r, on_g, path):
    """Comparison figure (4 rows):
       row 1: real 10 s | generated first 10 s            (side by side, same axes)
       row 2: generated recording, up to 60 s              (first 10 s shaded = the part compared with real)
       row 3: spike count per window: real | generated     (side by side, same axes)
       row 4: power spectrum | amplitude histogram         (real and generated overlaid)
    """
    import matplotlib
    matplotlib.use("Agg")                                    # draw to a file, no display needed
    import matplotlib.pyplot as plt
    from scipy import signal

    n10 = len(xr)                                            # samples in the real recording (10 s)
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.8, 1])
    ylim = (min(xr.min(), xg.min()) - 0.3, max(xr.max(), xg.max()) + 0.3)   # same y-range for fair comparison

    # row 1: the two 10 s recordings, side by side
    a = fig.add_subplot(gs[0, 0]); a.plot(tr, xr, lw=.5, color="C0"); a.set_title("Real data (10 s)")
    b = fig.add_subplot(gs[0, 1], sharey=a); b.plot(tg[:n10], xg[:n10], lw=.5, color="C1")
    b.set_title("Generated data (first 10 s)")
    a.set_ylim(*ylim)
    for ax in (a, b):
        ax.set_xlabel("time (s)"); ax.set_xlim(0, 10)
    a.set_ylabel("voltage")

    # row 2: the long generated recording
    c = fig.add_subplot(gs[1, :]); c.plot(tg, xg, lw=.4, color="C1")
    c.axvspan(0, n10 / fs, color="grey", alpha=.15, label="first 10 s (compared with real data)")
    c.set_title(f"Generated data, full length ({len(xg)/fs:g} s)"); c.set_xlabel("time (s)"); c.set_ylabel("voltage")
    c.set_xlim(0, len(xg) / fs); c.legend(loc="upper right")

    # row 3: spike counts per window, real vs generated, same axes
    nw = int(n10 / fs // win)
    from .model import window_counts
    cr = window_counts(on_r, len(xr), fs, win)[:nw]
    cg = window_counts(on_g, len(xg), fs, win)[:nw]
    top = max(cr.max(), cg.max()) + 1
    d = fig.add_subplot(gs[2, 0]); d.bar(np.arange(nw) * win, cr, width=win * .9, align="edge", color="C0")
    d.set_title(f"Real: spikes per {win:g} s window"); d.set_ylim(0, top)
    e = fig.add_subplot(gs[2, 1], sharey=d); e.bar(np.arange(nw) * win, cg, width=win * .9, align="edge", color="C1")
    e.set_title(f"Generated (first 10 s): spikes per {win:g} s window")
    for ax in (d, e):
        ax.set_xlabel("window start (s)")
    d.set_ylabel("spike count")

    # row 4: distributions (overlaid on purpose, so differences are easy to see)
    f, pa = signal.welch(xr, fs, nperseg=512); _, pb = signal.welch(xg, fs, nperseg=512)
    g = fig.add_subplot(gs[3, 0]); g.loglog(f[1:], pa[1:], label="real"); g.loglog(f[1:], pb[1:], label="generated")
    g.set_title("Power spectrum (Welch)"); g.set_xlabel("frequency (Hz)"); g.legend()
    bins = np.linspace(-2, 7, 120)
    h = fig.add_subplot(gs[3, 1]); h.hist(xr, bins, density=True, alpha=.5, label="real")
    h.hist(xg, bins, density=True, alpha=.5, label="generated"); h.set_yscale("log")
    h.set_title("Amplitude histogram"); h.set_xlabel("voltage"); h.legend()

    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
