"""Command line interface:  python -m Src <fit | generate | evaluate> [options]"""
import argparse
import os

from .evaluate import run
from .fit import fit
from .generate import generate
from .io import load_npz, save_npz
from .model import Params

# project root = the folder that contains Src/, data/, params/, results/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = lambda *p: os.path.join(ROOT, *p)                        # build a path inside the project


def main(argv=None):
    ap = argparse.ArgumentParser(prog="Src")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- fit: extract parameters (+ 0.5 s spike counts) from the real recording ---
    a = sub.add_parser("fit", help="extract parameters from the real recording")
    a.add_argument("--data", default=D("data", "spike_data_2.npz"))
    a.add_argument("--out", default=D("params", "fitted_params.json"))
    a.add_argument("--window", type=float, default=0.5, help="spike-count window in seconds")

    # --- generate: write a new synthetic recording ---
    g = sub.add_parser("generate", help="generate a new recording")
    g.add_argument("--duration", type=float, default=60.0, help="length in seconds")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--spike-mode", choices=["window", "poisson"], default="window",
                   help="window = use real 0.5 s spike counts (default); poisson = random times at the average rate")
    g.add_argument("--params", default=D("params", "fitted_params.json"))
    g.add_argument("--out", default=D("results", "generated_data.npz"))

    # --- evaluate: print the score table ---
    e = sub.add_parser("evaluate", help="print the accuracy score table")
    e.add_argument("--real", default=D("data", "spike_data_2.npz"))
    e.add_argument("--params", default=D("params", "fitted_params.json"))
    e.add_argument("--generated", default=None, help="score this .npz instead of generating fresh data")
    e.add_argument("--duration", type=float, default=60.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--spike-mode", choices=["window", "poisson"], default="window")
    e.add_argument("--seeds", type=int, default=0, help="also report the score over N seeds")
    e.add_argument("--plot", default=None, help="write the comparison figure to this PNG path")
    s = ap.parse_args(argv)

    if s.cmd == "fit":
        t, x = load_npz(s.data)
        p, info = fit(t, x, window_s=s.window)
        p.save(s.out, {"n_spikes": info["n_spikes"], "std_errors": info["std_errors"]})
        print(open(s.out).read())
    elif s.cmd == "generate":
        t, x, on = generate(s.duration, Params.load(s.params), seed=s.seed, spike_mode=s.spike_mode)
        save_npz(s.out, t, x)
        print(f"wrote {s.out}: {len(x)} samples, {len(on)} spikes")
    else:
        run(s.real, s.params, s.generated, s.duration, s.seed, s.seeds, s.plot, s.spike_mode)
