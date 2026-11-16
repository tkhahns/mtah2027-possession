"""Production wrapper for the Torch set-temporal and semi-Markov model."""

import os

import numpy as np

from . import learned_timing_pipeline, segmental_decoder, set_temporal

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(ROOT, "src", "models", "set_temporal.pt")


def resolve_device(requested="auto"):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(requested)


def load_checkpoint(path=DEFAULT_MODEL, device="auto"):
    """Load on CUDA when available, with a CPU-safe map location otherwise."""
    import torch

    device = resolve_device(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = set_temporal.build_model()
    model.load_state_dict(checkpoint["state_dict"])
    checkpoint = dict(checkpoint)
    checkpoint["model"] = model.to(device).eval()
    checkpoint["device"] = device
    return checkpoint


def apply(base, trk, state_probability, boundary_probability, selection, duration=None):
    """Apply the selected decoder to frozen base labels and neural probabilities."""
    mode = selection.get("mode", "baseline")
    parameters = selection.get("parameters") or {}
    if mode == "baseline":
        return np.asarray(base).copy()
    if mode == "boundary":
        return set_temporal.refine(base, state_probability, boundary_probability, **parameters)
    if mode == "segmental":
        if duration is None:
            raise ValueError("segmental decoding requires duration statistics")
        return segmental_decoder.decode(
            base, trk, state_probability, boundary_probability, duration, **parameters)
    raise ValueError("unknown set-temporal decoder mode: {}".format(mode))


def predict(trk, checkpoint=DEFAULT_MODEL, device="auto"):
    """Predict H/A/D using the frozen base, set encoder and selected segment decoder."""
    if isinstance(checkpoint, (str, os.PathLike)):
        checkpoint = load_checkpoint(checkpoint, device)
    base = learned_timing_pipeline.predict(trk)
    state, boundary, index = set_temporal.predict_probabilities(
        checkpoint["model"], trk, device=checkpoint.get("device", resolve_device(device)),
        stride=int(checkpoint.get("stride", set_temporal.STRIDE)))
    state = set_temporal.upsample(state, index, len(base))
    boundary = set_temporal.upsample(boundary, index, len(base))
    return apply(base, trk, state, boundary, checkpoint["configuration"],
                 checkpoint.get("duration"))
