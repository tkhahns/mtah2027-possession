"""Conservative restart decoder anchored by hard dead-ball evidence.

Medium-confidence learned state may extend or join hard dead intervals. A learned-only interval
needs higher confidence, sufficient duration, and a stopped-to-moving transition near its end.
"""

import os

import numpy as np

from . import hybrid_pipeline, restart_state

EXTENSION_THRESHOLD = 0.80
STANDALONE_THRESHOLD = 0.85
MIN_STANDALONE_FRAMES = 100
MOTION_THRESHOLD = 0.25
MOTION_BEFORE = 25
MOTION_AFTER = 50


def motion_evidence(features):
    """0..1 stopped-to-moving evidence derived from contextual feature transitions."""
    features = np.asarray(features)
    column = {name: features[:, i] for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_after = column["ball_speed_after"] * 15.0
    ball_change = column["ball_speed_change"] * 15.0
    player_after = column["player_speed_after"] * 5.0
    player_change = column["player_speed_change"] * 5.0
    visible_change = column["visible_share_change"]
    ball = np.clip(ball_change / 3.0, 0.0, 1.0) * np.clip(ball_after / 3.0, 0.0, 1.0)
    player = (np.clip(player_change / 1.0, 0.0, 1.0) *
              np.clip(player_after / 1.5, 0.0, 1.0))
    visibility = (np.clip(visible_change / 0.5, 0.0, 1.0) *
                  np.clip(ball_after / 3.0, 0.0, 1.0))
    return np.maximum.reduce([ball, player, 0.5 * visibility]).astype(np.float32)


def _runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))


def constrained_dead(hard, probability, evidence, extension_threshold=EXTENSION_THRESHOLD,
                     standalone_threshold=STANDALONE_THRESHOLD,
                     min_standalone_frames=MIN_STANDALONE_FRAMES,
                     motion_threshold=MOTION_THRESHOLD):
    """Combine anchored extensions and strongly supported learned-only restart intervals."""
    hard = np.asarray(hard, dtype=bool)
    probability = np.asarray(probability)
    evidence = np.asarray(evidence)
    dead = hard.copy()

    # A connected medium-confidence component is retained only when it contains hard evidence.
    supported = hard | (probability >= extension_threshold)
    for start, stop in _runs(supported):
        if hard[start:stop].any():
            dead[start:stop] = True

    # Unanchored components need stronger state confidence and restart motion at their end.
    standalone = (probability >= standalone_threshold) & ~dead
    for start, stop in _runs(standalone):
        if stop - start < min_standalone_frames:
            continue
        lo, hi = max(start, stop - MOTION_BEFORE), min(len(dead), stop + MOTION_AFTER)
        if hi > lo and np.nanmax(evidence[lo:hi]) >= motion_threshold:
            dead[start:stop] = True
    return dead


def predict_from_probabilities(trk, probability, features, window=50, smooth=restart_state.WINDOW,
                               **decoder):
    probability = restart_state._window_mean(probability, smooth, smooth) if smooth else probability
    evidence = motion_evidence(features)
    dead = constrained_dead(hybrid_pipeline.dead_ball(trk), probability, evidence, **decoder)
    return hybrid_pipeline.labels_from_codes(hybrid_pipeline.nearest_team(trk), dead, window)


def predict(trk, model=restart_state.DEFAULT_MODEL, window=50, **decoder):
    model = restart_state.load_model(model) if isinstance(model, (str, os.PathLike)) else model
    features = restart_state.frame_features(trk)
    probability = restart_state.probabilities(features, model)
    return predict_from_probabilities(trk, probability, features, window=window, **decoder)
