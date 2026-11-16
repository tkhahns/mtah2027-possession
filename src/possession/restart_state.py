"""Spatial context features and a learned dead/live restart-state classifier.

The hard out-of-pitch rule remains authoritative.  This model adds contextual evidence for dead
intervals that tracking geometry alone misses, such as fouls and centrally taken free kicks.
Training uses Torch on complete public matches; the exported checkpoint runs with NumPy.
"""

import json
import os

import numpy as np

from . import hybrid_pipeline, spatial_shape

FPS = 25
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(ROOT, "src", "models", "restart_state.json")
DEFAULT_THRESHOLD = 1.0
WINDOW = 12

BASE_FEATURE_NAMES = (
    "centroid_x", "centroid_y", "spread_x", "spread_y", "linearity",
    "visible_share", "player_speed",
    "home_centroid_x", "home_centroid_y", "away_centroid_x", "away_centroid_y",
    "home_spread", "away_spread", "home_linearity", "away_linearity", "team_separation",
    "local_centroid_x", "local_centroid_y", "local_spread", "local_linearity",
    "local_count_share", "local_team_balance",
    "ball_x", "ball_y", "ball_visible", "ball_boundary_distance", "ball_speed",
    "last_ball_x", "last_ball_y", "last_boundary_distance", "ball_missing_age",
    "centroid_motion",
)
TEMPORAL_SIGNALS = ("player_speed", "global_spread", "local_count", "ball_speed", "visible_share")
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(
    "{}_{}".format(name, moment)
    for name in TEMPORAL_SIGNALS for moment in ("before", "after", "change")
)


def _window_mean(values, before, after):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    valid = np.isfinite(values)
    total = np.concatenate([np.zeros((1, values.shape[1])),
                            np.cumsum(np.where(valid, values, 0.0), axis=0)])
    count = np.concatenate([np.zeros((1, values.shape[1]), dtype=np.int64),
                            np.cumsum(valid, axis=0)])
    index = np.arange(len(values))
    lo = np.maximum(0, index - before)
    hi = np.minimum(len(values), index + after + 1)
    denominator = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator > 0, (total[hi] - total[lo]) / denominator, np.nan)
    return result[:, 0] if result.shape[1] == 1 else result


def _last_observed(ball):
    visible = np.isfinite(ball[:, :2]).all(axis=1)
    last = np.maximum.accumulate(np.where(visible, np.arange(len(ball)), -1))
    xy = np.full((len(ball), 2), np.nan, np.float32)
    known = last >= 0
    xy[known] = ball[last[known], :2]
    age = np.where(known, (np.arange(len(ball)) - last) / FPS, 10.0)
    return xy, np.minimum(age, 10.0), visible


def _ball_speed(ball):
    ball = np.asarray(ball, dtype=np.float32)
    supplied = ball[:, 2].copy() if ball.shape[1] > 2 else np.full(len(ball), np.nan, np.float32)
    xy = ball[:, :2]
    valid = np.isfinite(xy[1:]).all(axis=1) & np.isfinite(xy[:-1]).all(axis=1)
    derived = np.full(len(ball), np.nan, np.float32)
    derived[1:][valid] = np.linalg.norm(xy[1:][valid] - xy[:-1][valid], axis=1) * FPS
    return np.where(np.isfinite(supplied), supplied, derived)


def frame_features(trk, window=WINDOW, shape=None):
    """Return finite, normalized per-frame context for dead/live classification."""
    shape = spatial_shape.frame_features(trk) if shape is None else shape
    ball = np.asarray(trk["ball"], dtype=np.float32)
    length, width = map(float, trk.get("pitch", (105.0, 68.0)))
    last_ball, missing_age, ball_visible = _last_observed(ball)

    global_spread = np.sqrt(np.maximum(shape["variance"], 0.0))
    team_spread = np.sqrt(np.maximum(np.nansum(shape["team_variance"], axis=2), 0.0))
    local_spread = np.sqrt(np.maximum(np.nansum(shape["local_variance"], axis=1), 0.0))
    team_separation = np.linalg.norm(
        shape["team_centroid"][:, 0] - shape["team_centroid"][:, 1], axis=1)
    centroid_motion = np.r_[np.nan, np.linalg.norm(np.diff(shape["centroid"], axis=0), axis=1) * FPS]
    ball_speed = _ball_speed(ball)

    boundary = np.minimum.reduce([ball[:, 0], length - ball[:, 0],
                                  ball[:, 1], width - ball[:, 1]])
    last_boundary = np.minimum.reduce([last_ball[:, 0], length - last_ball[:, 0],
                                       last_ball[:, 1], width - last_ball[:, 1]])
    ball_xy = np.where(ball_visible[:, None], ball[:, :2], [length / 2, width / 2])

    columns = [
        shape["centroid"][:, 0] / length, shape["centroid"][:, 1] / width,
        global_spread[:, 0] / length, global_spread[:, 1] / width, shape["linearity"],
        shape["visible_count"] / 22.0, shape["mean_speed"] / 5.0,
        shape["team_centroid"][:, 0, 0] / length, shape["team_centroid"][:, 0, 1] / width,
        shape["team_centroid"][:, 1, 0] / length, shape["team_centroid"][:, 1, 1] / width,
        team_spread[:, 0] / 30.0, team_spread[:, 1] / 30.0,
        shape["team_linearity"][:, 0], shape["team_linearity"][:, 1], team_separation / 30.0,
        shape["local_centroid"][:, 0] / length, shape["local_centroid"][:, 1] / width,
        local_spread / 15.0, shape["local_linearity"], shape["local_count"] / 10.0,
        (shape["local_team_count"][:, 0] - shape["local_team_count"][:, 1]) / 10.0,
        ball_xy[:, 0] / length, ball_xy[:, 1] / width, ball_visible.astype(np.float32),
        boundary / 15.0, ball_speed / 15.0,
        last_ball[:, 0] / length, last_ball[:, 1] / width, last_boundary / 15.0,
        missing_age / 5.0, centroid_motion / 5.0,
    ]
    signals = (
        shape["mean_speed"] / 5.0,
        np.sqrt(np.maximum(np.nansum(shape["variance"], axis=1), 0.0)) / 30.0,
        shape["local_count"] / 10.0,
        ball_speed / 15.0,
        shape["visible_count"] / 22.0,
    )
    for values in signals:
        before = _window_mean(values, window, 0)
        after = _window_mean(values, 0, window)
        columns.extend([before, after, after - before])
    features = np.stack(columns, axis=1).astype(np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=4.0, neginf=-4.0)


def load_model(path=DEFAULT_MODEL):
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("restart-state checkpoint feature layout does not match the code")
    return model


def probabilities(features, model):
    """NumPy inference for the Torch-trained three-layer MLP."""
    x = ((np.asarray(features, np.float32) - np.asarray(model["mean"], np.float32)) /
         np.asarray(model["std"], np.float32))
    first = np.maximum(0.0, x @ np.asarray(model["weight1"], np.float32).T +
                       np.asarray(model["bias1"], np.float32))
    second = np.maximum(0.0, first @ np.asarray(model["weight2"], np.float32).T +
                        np.asarray(model["bias2"], np.float32))
    logits = second @ np.asarray(model["weight3"], np.float32) + float(model["bias3"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def dead_mask(trk, model=DEFAULT_MODEL, threshold=None, smooth=WINDOW, features=None):
    """Hard dead-ball evidence plus high-confidence learned dead-state intervals."""
    model = load_model(model) if isinstance(model, (str, os.PathLike)) else model
    threshold = float(model.get("threshold", DEFAULT_THRESHOLD) if threshold is None else threshold)
    features = frame_features(trk) if features is None else features
    probability = probabilities(features, model)
    probability = _window_mean(probability, smooth, smooth) if smooth else probability
    return hybrid_pipeline.dead_ball(trk) | (probability >= threshold)
