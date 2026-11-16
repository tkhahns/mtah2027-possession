"""Dead-ball/carry possession pipeline with learned spatial restart state."""

import os

import numpy as np

from . import hybrid_pipeline, restart_state


def combine_dead(hard, probability, threshold, min_dead_frames=0):
    """Combine hard and learned evidence, dropping short learned-only dead components."""
    hard = np.asarray(hard, dtype=bool)
    dead = hard | (np.asarray(probability) >= threshold)
    if min_dead_frames <= 1 or not dead.any():
        return dead
    change = np.diff(np.r_[False, dead, False].astype(np.int8))
    for start, stop in zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)):
        if stop - start < min_dead_frames and not hard[start:stop].any():
            dead[start:stop] = False
    return dead


def predict_from_probabilities(trk, probability, threshold, window=50, smooth=restart_state.WINDOW,
                               min_dead_frames=0):
    probability = restart_state._window_mean(probability, smooth, smooth) if smooth else probability
    dead = combine_dead(hybrid_pipeline.dead_ball(trk), probability, threshold, min_dead_frames)
    return hybrid_pipeline.labels_from_codes(hybrid_pipeline.nearest_team(trk), dead, window)


def predict(trk, model=restart_state.DEFAULT_MODEL, threshold=None, window=50):
    model = restart_state.load_model(model) if isinstance(model, (str, os.PathLike)) else model
    threshold = float(model["threshold"] if threshold is None else threshold)
    features = restart_state.frame_features(trk)
    probability = restart_state.probabilities(features, model)
    return predict_from_probabilities(trk, probability, threshold, window,
                                      min_dead_frames=int(model.get("min_dead_frames", 0)))
