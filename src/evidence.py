"""Per-frame evidence from team shape and velocity: who has the ball, and is it dead?

    feats = team_shape.frame_features(trk)
    ev = evidence.evidence(trk, feats)
    ev["team"]   # (n,) log-odds that home (not away) has the ball, if it is live
    ev["dead"]   # (n,) log-odds that the ball is dead, from player motion only
    ev["oob"]    # (n,) bool: the ball's last sighting is off the pitch

Two small logistic models with frozen weights (shape_evidence.json), trained on Metrica's public
sample games only (hidden-ball and messy-tracking test sets), never on our match.

Why these inputs:

- The team in possession stretches the pitch, the defending team stays compact (team width).
  This needs no ball, so it still works on the many frames where the ball is lost.
- The team in possession moves forward (velocity along its attacking direction).
- Near the ball, a player within 1.5 m with no opponent within 2.5 m and a slow ball is control.
- The side closing in on the ball (moving towards it) is usually the one defending.
- The ball is dead when players stand or walk: share of slow players, mean speed, the speed of
  the players near the ball and of the whole block, over the seconds before and after.
- The dead model never uses whether the ball is seen: where the ball's position is unknown the
  near-ball speed falls back to every player's speed and the closing gap to 0.
"""

import json
import os

import numpy as np

import team_shape
from config import AWAY, HOME

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shape_evidence.json")

TEAM_INPUTS = ("width_diff", "length_diff", "forward_diff", "nearest_diff_slow",
               "nearest_diff_fast", "control_now", "control_before", "control_after",
               "closing_diff_1s", "closing_diff_now")
DEAD_INPUTS = ("still_2s", "speed_2s", "still_before_5s", "still_after_5s",
               "speed_before_5s", "speed_after_5s",
               "near_speed_2s", "near_speed_before_5s", "near_speed_after_5s",
               "centroid_speed_2s", "centroid_speed_after_5s")

CONTROL_M, FREE_M, CONTROL_BALL_SPEED = 1.5, 2.5, 7.0    # m, m, m/s


def control(feats):
    """(n,) +1 where a home player controls the ball, -1 away, 0 neither or ball not seen."""
    seen = feats["ball_seen"]
    near = feats["nearest"]
    slow = np.nan_to_num(feats["ball_speed"], nan=0.0) < CONTROL_BALL_SPEED
    with np.errstate(invalid="ignore"):
        home = seen & slow & (near[:, HOME] < CONTROL_M) & (near[:, AWAY] > FREE_M)
        away = seen & slow & (near[:, AWAY] < CONTROL_M) & (near[:, HOME] > FREE_M)
    return home.astype(np.float32) - away.astype(np.float32)


def team_inputs(feats):
    """(n, len(TEAM_INPUTS)) home-minus-away inputs; 0 where there is nothing to say."""
    wm = team_shape.window_mean
    spread, near = feats["spread"], feats["nearest"]
    with np.errstate(invalid="ignore"):
        gap = near[:, AWAY] - near[:, HOME]
    fast = np.where(feats["ball_seen"], np.clip(gap, -10, 10), np.nan)
    ctl = control(feats)
    closing = np.nan_to_num(feats["closing"][:, AWAY] - feats["closing"][:, HOME])   # no ball: 0
    cols = [wm(spread[:, HOME, 1] - spread[:, AWAY, 1], 25, 25),
            wm(spread[:, HOME, 0] - spread[:, AWAY, 0], 25, 25),
            wm(feats["forward"][:, HOME] - feats["forward"][:, AWAY], 25, 25),
            wm(np.clip(gap, -15, 15), 25, 25),
            wm(fast, 3, 3),
            wm(ctl, 5, 5), wm(ctl, 25, 0), wm(ctl, 0, 25),
            wm(closing, 25, 25), wm(closing, 5, 5)]
    return np.nan_to_num(np.stack(cols, axis=1)).astype(np.float32)


def dead_inputs(feats):
    """(n, len(DEAD_INPUTS)) player-motion inputs (no ball)."""
    wm = team_shape.window_mean
    team_speed = feats["speed"]
    ok = np.isfinite(team_speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        speed = np.where(ok, team_speed, 0.0).sum(axis=1) / ok.sum(axis=1)   # NaN if neither
    still = feats["still"]
    near = np.where(np.isfinite(feats["near_speed"]), feats["near_speed"], speed)  # no ball: everyone
    centroid = feats["centroid_speed"]
    cols = [wm(still, 50, 50), wm(speed, 50, 50), wm(still, 125, 0), wm(still, 0, 125),
            wm(speed, 125, 0), wm(speed, 0, 125),
            wm(near, 50, 50), wm(near, 125, 0), wm(near, 0, 125),
            wm(centroid, 50, 50), wm(centroid, 0, 125)]
    return np.nan_to_num(np.stack(cols, axis=1)).astype(np.float32)


def out_of_pitch(trk):
    """(n,) True while the ball's latest sighting is beyond a line (stays True while it is lost)."""
    ball = np.asarray(trk["ball"], dtype=np.float32)[:, :2]
    seen = np.isfinite(ball).all(axis=1)
    last = np.maximum.accumulate(np.where(seen, np.arange(len(seen)), -1))
    length, width = team_shape.pitch_of(trk)
    x, y = ball[last, 0], ball[last, 1]
    with np.errstate(invalid="ignore"):
        return (last >= 0) & ((x < 0) | (x > length) | (y < 0) | (y > width))


def logodds(X, model):
    X = np.asarray(X, dtype=np.float64)
    z = (X - np.asarray(model["mean"])) / np.asarray(model["scale"])
    return (model["bias"] + z @ np.asarray(model["weights"])).astype(np.float32)


def load_models(path=MODEL_PATH):
    with open(path) as handle:
        models = json.load(handle)
    for name, inputs in (("team", TEAM_INPUTS), ("dead", DEAD_INPUTS)):
        if tuple(models[name]["inputs"]) != inputs:
            raise ValueError("{}: the {} model was trained on other inputs".format(path, name))
    return models


def evidence(trk, feats, models=None):
    """{"team", "dead", "oob"} per frame (see the module docstring)."""
    models = load_models() if models is None else models
    return {"team": logodds(team_inputs(feats), models["team"]),
            "dead": logodds(dead_inputs(feats), models["dead"]),
            "oob": out_of_pitch(trk)}
