"""Src: fit and regenerate a neural-style voltage recording.

The recording is modelled as   drift (sine)  +  spikes (exponential, random times)  +  white Gaussian noise.
Public API (everything else is internal):
    load_npz / save_npz  - read / write files shaped like the original (keys: t, raw_data)
    Params               - the fitted model parameters (stored in params/fitted_params.json)
    fit                  - extract all parameters from a recording
    detect_spikes        - find spike onset samples in a recording
    generate             - create a new recording of any length
"""
from .model import Params
from .io import load_npz, save_npz
from .fit import fit, detect_spikes
from .generate import generate

__all__ = ["Params", "load_npz", "save_npz", "fit", "detect_spikes", "generate"]
